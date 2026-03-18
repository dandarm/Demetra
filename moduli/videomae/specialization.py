import numpy as np
import os
import datetime
import time
import argparse
import math
from pathlib import Path
from PIL import Image
import json
import torch
import torch.nn.functional as F
import torch.distributed as dist

from utils import multiple_pretrain_samples_collate
from functools import partial

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_pretrain_samples_collate, setup_for_distributed
from optim_factory import LayerDecayValueAssigner, create_optimizer, create_split_adamw_optimizer
#from dataset import build_pretraining_dataset
from utils import get_model
from engine_for_pretraining import train_one_epoch, test
from arguments import prepare_args, Args  # NON TOGLIERE: serve a torch.load per caricare il mio modello addestrato
from dataset.data_manager import DataManager


utils.suppress_transformers_pytree_warning()



def _clean_param_name(name):
    prefixes = ("module.", "_orig_mod.", "backbone.")
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if name.startswith(p):
                name = name[len(p):]
                changed = True
    return name


def _layer_key_for_param(name):
    name = _clean_param_name(name)
    parts = name.split(".")
    if name.startswith("encoder.blocks.") and len(parts) > 2 and parts[2].isdigit():
        return f"encoder.blocks.{parts[2]}"
    if name.startswith("decoder.blocks.") and len(parts) > 2 and parts[2].isdigit():
        return f"decoder.blocks.{parts[2]}"
    if name.startswith("encoder.patch_embed"):
        return "encoder.patch_embed"
    if name.startswith("encoder.norm"):
        return "encoder.norm"
    if name.startswith("decoder.norm"):
        return "decoder.norm"
    if name.startswith("decoder.head"):
        return "decoder.head"
    if name.startswith("encoder_to_decoder"):
        return "encoder_to_decoder"
    if name in ("mask_token", "pos_embed"):
        return name
    return parts[0] if parts else name


def _capture_reference_params(model):
    reference = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            reference[name] = param.detach().to(dtype=torch.float32, device="cpu").clone()
    return reference


def _compute_weight_drift_by_layer(model, reference_params):
    layer_stats = {}
    overall_diff_sq = 0.0
    overall_ref_sq = 0.0
    missing_params = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            ref = reference_params.get(name, None)
            if ref is None or ref.shape != param.shape:
                missing_params += 1
                continue

            current = param.detach().to(dtype=torch.float32, device="cpu")
            diff = current - ref
            diff_sq = float(torch.sum(diff * diff).item())
            ref_sq = float(torch.sum(ref * ref).item())

            layer_key = _layer_key_for_param(name)
            if layer_key not in layer_stats:
                layer_stats[layer_key] = {"diff_sq": 0.0, "ref_sq": 0.0}
            layer_stats[layer_key]["diff_sq"] += diff_sq
            layer_stats[layer_key]["ref_sq"] += ref_sq

            overall_diff_sq += diff_sq
            overall_ref_sq += ref_sq

    layer_rel = {}
    for layer_key, stats in layer_stats.items():
        denom = max(stats["ref_sq"] ** 0.5, 1e-12)
        layer_rel[layer_key] = (stats["diff_sq"] ** 0.5) / denom

    overall_rel = (overall_diff_sq ** 0.5) / max(overall_ref_sq ** 0.5, 1e-12)
    return {
        "overall_rel_l2": overall_rel,
        "layer_rel_l2": dict(sorted(layer_rel.items())),
        "missing_params": missing_params,
    }


def _print_weight_drift_summary(drift_stats, top_k=10):
    overall = drift_stats.get("overall_rel_l2", 0.0)
    missing = drift_stats.get("missing_params", 0)
    layer_rel = drift_stats.get("layer_rel_l2", {})
    print(
        f"[DRIFT] overall_rel_l2={overall:.6e}, "
        f"tracked_layers={len(layer_rel)}, missing_params={missing}"
    )
    top_layers = sorted(layer_rel.items(), key=lambda x: x[1], reverse=True)[:top_k]
    for layer_key, rel_value in top_layers:
        print(f"[DRIFT][TOP] {layer_key} rel_l2={rel_value:.6e}")


def _set_encoder_trainable(model, trainable):
    if not hasattr(model, "encoder"):
        return
    for param in model.encoder.parameters():
        param.requires_grad = bool(trainable)


def _model_non_finite_summary(model, max_examples=10):
    bad_tensors = 0
    bad_elems = 0
    examples = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            t = param.detach()
            bad = (~torch.isfinite(t)).sum().item()
            if bad:
                bad_tensors += 1
                bad_elems += int(bad)
                if len(examples) < max_examples:
                    examples.append((name, int(bad), tuple(t.shape)))
    return bad_tensors, bad_elems, examples


def launch_specialization_training(terminal_args):
    args = prepare_args(machine=terminal_args.on)
    utils.apply_compile_override(args, getattr(terminal_args, "compile", "auto"))

    # Backward compatibility: historical pretraining config used `data_path`
    # for train CSV, while DataManager expects `train_path`.
    if not getattr(args, "train_path", None):
        args.train_path = args.data_path
    if not getattr(args, "test_path", None):
        args.test_path = args.data_path
    utils.print_perf_config(
        args,
        extra_keys=[
            "perf_profile_every",
            "enable_weight_drift_logging",
            "use_split_encoder_decoder_lr",
            "encoder_lr",
            "decoder_lr",
            "freeze_encoder_epochs",
        ],
    )
    utils.resolve_args_paths(
        args,
        __file__,
        ["train_path", "test_path", "output_dir", "log_dir", "init_ckpt", "resume"],
    )

    # Prefer latest training checkpoint for resume when available.
    if args.auto_resume and not args.resume:
        args.resume = utils.find_latest_numbered_checkpoint(args.output_dir, "checkpoint-*.pth")

    # If resuming, initialize model from that checkpoint too (same weights path).
    if args.resume and Path(args.resume).exists():
        print(f"[RESUME] Using training checkpoint: {args.resume}")
        args.init_ckpt = args.resume
    

    #utils.init_distributed_mode(args)
    #local_rank = int(os.environ["LOCAL_RANK"])
    #torch.cuda.set_device(local_rank)
    #device = torch.device(f"cuda:{local_rank}")
    #print(device)
    #print(f"[rank {dist.get_rank()}] running on {torch.cuda.current_device()}")

    rank, local_rank, world_size, local_size, num_workers = utils.get_resources()
    #print(f"rank, local_rank, world_size, local_size, num_workers: {rank, local_rank, world_size, local_size, num_workers}")

    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)    
        args.distributed = True
    else:
        args.distributed = False   

    if args.device == 'cuda':
        torch.cuda.set_device(local_rank)
        args.gpu = local_rank
        args.world_size = world_size
        args.rank = rank
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        args.gpu = None
        args.world_size = 1
        args.rank = 0
    #print(device)

    utils.configure_cuda_runtime(args, device, rank)

    # logging
    if args.log_dir and not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    setup_for_distributed(rank == 0)

    # Optional control for torch.compile (Inductor) CUDA Graphs.
    # Must be configured before get_model() calls torch.compile.
    utils.configure_inductor_cudagraphs(args, rank)


    # LOAD MODEL
    pretrained_model = get_model(args)


    pretrained_model.to(device)  # rimane solo sul device generale o va messo nel local_rank?
    model_without_ddp = pretrained_model
    n_parameters = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
    #print("Model = %s" % str(model_without_ddp))
    #print('number of params: {} M'.format(n_parameters / 1e6))

    pretrained_model, model_without_ddp = utils.wrap_model_distributed(pretrained_model, args, rank)



    # LOAD DATASET (aligned with DataManager used in classification/tracking)
    patch_size = model_without_ddp.encoder.patch_embed.patch_size
    train_m = DataManager(
        is_train=True,
        args=args,
        type_t='unsupervised',
        patch_size=patch_size,
        world_size=world_size,
        rank=rank,
    )
    test_m = DataManager(
        is_train=False,
        args=args,
        type_t='unsupervised',
        patch_size=patch_size,
        world_size=world_size,
        rank=rank,
    )
    train_m.create_specialization_dataloader(args)
    test_m.create_specialization_dataloader(args)
    data_loader_train = train_m.data_loader
    data_loader_test = test_m.data_loader
    dataset_train = train_m.dataset


    # SET HYPER PARAMETERS
    num_tasks = utils.get_world_size()
    total_batch_size = args.batch_size * num_tasks
    # Use effective dataloader length (accounts for sampler/drop_last in DDP).
    num_training_steps_per_epoch = len(data_loader_train)
    if num_training_steps_per_epoch <= 0:
        raise ValueError(
            "num_training_steps_per_epoch=0: dataset troppo piccolo rispetto al batch effettivo. "
            f"dataset_len={len(dataset_train)}, total_batch_size={total_batch_size}, "
            f"batch_size={args.batch_size}, world_size={num_tasks}. "
            "Riduci batch_size o aumenta il numero di sample nel manifest."
        )
    use_split_encoder_decoder_lr = bool(
        getattr(args, "use_split_encoder_decoder_lr", False)
        and hasattr(model_without_ddp, "encoder")
        and hasattr(model_without_ddp, "decoder")
    )
    if use_split_encoder_decoder_lr:
        legacy_base_lr = float(args.lr)
        legacy_min_lr = float(args.min_lr)
        legacy_warmup_lr = float(args.warmup_lr)
        args.lr = float(args.decoder_lr)
        if legacy_base_lr <= 0:
            raise ValueError(f"args.lr must be > 0 for split LR scheduling, got {legacy_base_lr}")
        args.min_lr = args.lr * (legacy_min_lr / legacy_base_lr)
        args.warmup_lr = args.lr * (legacy_warmup_lr / legacy_base_lr)
        print(
            "Split encoder/decoder LR enabled: "
            f"encoder_lr={float(args.encoder_lr):.8f}, "
            f"decoder_lr={float(args.decoder_lr):.8f}, "
            f"scheduler_base_lr={args.lr:.8f}, min_lr={args.min_lr:.8f}, warmup_lr={args.warmup_lr:.8f}"
        )
        print("Linear LR scaling by total batch size disabled for split encoder/decoder LR mode.")
    else:
        # scale the lr
        args.lr = args.lr * total_batch_size / 256
        args.min_lr = args.min_lr * total_batch_size / 256
        args.warmup_lr = args.warmup_lr * total_batch_size / 256
    if args.warmup_epochs > 0 and args.warmup_lr <= 0:
        raise ValueError(
            f"warmup_lr deve essere > 0 quando warmup_epochs > 0 (attuale warmup_lr={args.warmup_lr})."
        )
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" %
          (total_batch_size * num_training_steps_per_epoch))


    assigner = None
    if (not use_split_encoder_decoder_lr) and hasattr(args, "layer_decay") and args.layer_decay is not None and args.layer_decay < 1.0:
        encoder_layers = None
        if hasattr(model_without_ddp, "encoder") and hasattr(model_without_ddp.encoder, "get_num_layers"):
            encoder_layers = model_without_ddp.encoder.get_num_layers()
        elif hasattr(model_without_ddp, "get_num_layers"):
            try:
                encoder_layers = model_without_ddp.get_num_layers()
            except Exception:
                encoder_layers = None
        if encoder_layers is not None and encoder_layers > 0:
            num_layers = int(encoder_layers) + 2
            scales = [args.layer_decay ** (num_layers - i - 1) for i in range(num_layers)]
            assigner = LayerDecayValueAssigner(scales)
            print(
                f"Layer decay enabled: layer_decay={args.layer_decay}, "
                f"encoder_layers={encoder_layers}, num_layers={num_layers}"
            )

    if use_split_encoder_decoder_lr:
        print(
            f"Split optimizer with branch-specific layer decay enabled: "
            f"layer_decay={getattr(args, 'layer_decay', None)}"
        )
        optimizer = create_split_adamw_optimizer(args, model_without_ddp)
    else:
        optimizer = create_optimizer(
            args,
            model_without_ddp,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None,
        )
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        start_warmup_value=args.warmup_lr,
        warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(args.weight_decay,
                                                args.weight_decay_end,
                                                args.epochs,
                                                num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" %
          (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args,
        model=pretrained_model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler)

    weight_reference_params = None
    if bool(getattr(args, "enable_weight_drift_logging", False)) and utils.is_main_process():
        print("Capturing reference weights for per-layer drift logging...")
        weight_reference_params = _capture_reference_params(model_without_ddp)
        print(f"[DRIFT] reference tensors captured: {len(weight_reference_params)}")
    # Rolling checkpoint logic state:
    # save only when current test loss improves over the previous evaluated test loss.
    prev_test_loss = None
    last_improved_ckpt_path = None
    if getattr(args, "resume", ""):
        try:
            resumed = torch.load(args.resume, map_location="cpu")
            if isinstance(resumed, dict) and "prev_test_loss" in resumed:
                prev_test_loss = float(resumed["prev_test_loss"])
            if str(args.resume).endswith(".pth"):
                last_improved_ckpt_path = str(args.resume)
        except Exception:
            pass
    torch.cuda.empty_cache()

    # SET THE LOGGING
    global_rank = utils.get_rank()
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    
    setup_for_distributed(rank == 0)

    ############################################
    ########################### START TRAINING #
    ############################################
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        freeze_encoder_epochs = int(getattr(args, "freeze_encoder_epochs", 0) or 0)
        encoder_trainable = epoch >= freeze_encoder_epochs
        _set_encoder_trainable(model_without_ddp, encoder_trainable)
        if utils.is_main_process():
            if epoch == args.start_epoch:
                state_label = "trainable" if encoder_trainable else "frozen"
                print(
                    f"Encoder freeze schedule: freeze_encoder_epochs={freeze_encoder_epochs}, "
                    f"encoder is {state_label} at epoch {epoch}"
                )
            elif freeze_encoder_epochs > 0 and epoch == freeze_encoder_epochs:
                print(f"Encoder unfrozen at epoch {epoch}")
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            pretrained_model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            patch_size=patch_size[0],
            normlize_target=args.normlize_target,
            amp_dtype=getattr(args, "amp_dtype", "fp16"),
            perf_profile_every=int(getattr(args, "perf_profile_every", 0) or 0),
            perf_profile_warmup=int(getattr(args, "perf_profile_warmup", 20) or 0))

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,
            #'n_parameters': n_parameters
        }

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if epoch % args.testing_epochs == 0:
            test_stats = test(pretrained_model, data_loader_test, device, epoch,
                        patch_size=patch_size[0], normlize_target=args.normlize_target,
                        log_writer=log_writer, amp_dtype=getattr(args, "amp_dtype", "fp16"))
            test_log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch}  #, 'n_parameters': n_parameters}
            if (
                bool(getattr(args, "enable_weight_drift_logging", False))
                and utils.is_main_process()
                and weight_reference_params is not None
            ):
                drift_stats = _compute_weight_drift_by_layer(model_without_ddp, weight_reference_params)
                _print_weight_drift_summary(drift_stats, top_k=10)
                test_log_stats["weight_drift_overall_rel_l2"] = float(drift_stats["overall_rel_l2"])
                test_log_stats["weight_drift_missing_params"] = int(drift_stats["missing_params"])
                test_log_stats["weight_drift_by_layer_rel_l2"] = drift_stats["layer_rel_l2"]
            test_loss = test_stats.get("loss", None)
            improved = False
            rng_state_payload = None
            if test_loss is not None:
                improved = (prev_test_loss is None) or (float(test_loss) < float(prev_test_loss))
                if improved:
                    # All ranks participate so DDP resume can restore per-rank RNG streams.
                    local_rng_state = utils.capture_rng_state()
                    rng_state_payload = utils.gather_rng_state_all_ranks(local_rng_state)

            if args.output_dir and utils.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(test_log_stats) + "\n")

                # Save checkpoint only if current test loss improves over previous test loss.
                # Skip save if test_loss or model weights are non-finite.
                if test_loss is not None and improved:
                    finite_test_loss = math.isfinite(float(test_loss))
                    bad_tensors, bad_elems, bad_examples = _model_non_finite_summary(model_without_ddp, max_examples=5)
                    finite_weights = (bad_tensors == 0)
                    if not finite_test_loss or not finite_weights:
                        print(
                            "[CKPT][SKIP] Non-finite state detected, checkpoint not saved: "
                            f"test_loss={test_loss}, finite_test_loss={finite_test_loss}, "
                            f"bad_tensors={bad_tensors}, bad_elems={bad_elems}, "
                            f"examples={bad_examples}"
                        )
                        # Do not overwrite previous best with a corrupted checkpoint.
                        if not finite_test_loss:
                            test_log_stats["checkpoint_skipped_non_finite_test_loss"] = True
                        if not finite_weights:
                            test_log_stats["checkpoint_skipped_non_finite_weights"] = True
                    else:
                        ckpt_dir = Path(args.output_dir)
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        ckpt_path = ckpt_dir / f"checkpoint-{epoch}.pth"
                        to_save = {
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'scaler': loss_scaler.state_dict(),
                            'args': args.__dict__,
                            'prev_test_loss': float(test_loss),
                        }
                        if isinstance(rng_state_payload, list):
                            to_save['rng_state_all_ranks'] = rng_state_payload
                        elif rng_state_payload is not None:
                            to_save['rng_state'] = rng_state_payload
                        torch.save(to_save, ckpt_path)
                        if last_improved_ckpt_path is not None and last_improved_ckpt_path != str(ckpt_path):
                            try:
                                old_path = Path(last_improved_ckpt_path)
                                if old_path.exists():
                                    old_path.unlink()
                            except Exception:
                                pass
                        last_improved_ckpt_path = str(ckpt_path)
                        print(
                            f"[CKPT] Saved improved checkpoint: {ckpt_path} "
                            f"(test_loss {test_loss:.6f}, prev {prev_test_loss})"
                        )
            if test_loss is not None:
                prev_test_loss = float(test_loss)




    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Lancia il training unsupervised di specialization',
        add_help=False)
    parser.add_argument('--on',
        type=str,
        default='leonardo',
        help='[ewc, leonardo]'
    )
    parser.add_argument('--compile',
        type=str,
        default='auto',
        choices=['auto', 'on', 'off'],
        help='Override compile_model: auto=usa arguments.py, on=True, off=False'
    )
    args = parser.parse_args()
    launch_specialization_training(args)
