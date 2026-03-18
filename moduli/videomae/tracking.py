"""Entry point for cyclone centre tracking training."""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from arguments import prepare_tracking_args
from optim_factory import create_optimizer
from dataset.data_manager import DataManager
from engine_for_tracking import train_one_epoch, evaluate
from models.tracking_model import create_tracking_model
from models.modeling_finetune import load_checkpoint as load_finetune_checkpoint
import utils


utils.suppress_transformers_pytree_warning()


def _format_log_value(key: str, value):
    """Keep learning-rate precision while rounding other floats for logs."""
    if not isinstance(value, float):
        return value
    if "lr" in key.lower():
        return float(f"{value:.8g}")
    return round(value, 4)


def _summarize_tracking_dataset(split_name: str, dataset, data_loader, rank: int) -> None:
    """Print a compact summary of tracking dataset availability."""
    if rank != 0:
        return

    dataset_rows = len(dataset) if dataset is not None else 0
    batches_per_rank = len(data_loader) if data_loader is not None else 0

    resolved_paths = getattr(dataset, "_resolved_paths", None)
    if isinstance(resolved_paths, list):
        total_paths = len(resolved_paths)
        existing_dirs = 0
        dirs_with_frames = 0
        for p in resolved_paths:
            if not isinstance(p, str):
                continue
            if os.path.isdir(p):
                existing_dirs += 1
                try:
                    has_img = any(name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
                                  for name in os.listdir(p))
                except Exception:
                    has_img = False
                if has_img:
                    dirs_with_frames += 1
    else:
        total_paths = dataset_rows
        existing_dirs = -1
        dirs_with_frames = -1

    print(
        f"[DATASET][{split_name}] rows={dataset_rows} | batches_per_rank={batches_per_rank} | "
        f"paths={total_paths} | existing_dirs={existing_dirs} | dirs_with_frames={dirs_with_frames}"
    )


def _resume_from_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    loss_scaler=None,
    rank: int = 0,
) -> Tuple[Optional[int], Optional[float]]:
    """Restore model (and optimizer) state from a checkpoint if possible."""
    if rank == 0:
        print(f"[INFO] Loading checkpoint for resume: {checkpoint_path}")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                state_dict = value
                if rank == 0:
                    print(f"[INFO] Using state dict stored under key '{key}'")
                break

    if state_dict is None:
        load_finetune_checkpoint(model, checkpoint_path, load_for_test_mode=True)
        if rank == 0:
            print("[WARN] Checkpoint did not contain explicit model weights; loaded via finetune helper.")
        resume_epoch = checkpoint["epoch"] if isinstance(checkpoint, dict) and "epoch" in checkpoint else None
        best_loss = checkpoint["best_loss"] if isinstance(checkpoint, dict) and "best_loss" in checkpoint else None
        return resume_epoch, best_loss

    cleaned_state_dict = {
        utils.strip_known_prefixes(key): value for key, value in state_dict.items()
    }

    model_state = model.state_dict()
    head_keys_to_drop = []
    for candidate in ("head.weight", "head.bias"):
        if candidate in cleaned_state_dict and candidate in model_state:
            if cleaned_state_dict[candidate].shape != model_state[candidate].shape:
                head_keys_to_drop.append(candidate)
    if head_keys_to_drop and rank == 0:
        print(f"[INFO] Dropping incompatible head parameters from checkpoint: {head_keys_to_drop}")
    for key in head_keys_to_drop:
        cleaned_state_dict.pop(key, None)

    load_result = utils.load_state_dict_with_alignment(
        model,
        cleaned_state_dict,
        rank=rank,
        strict=True,
    )
    if rank == 0:
        missing = sorted(load_result.missing_keys)
        unexpected = sorted(load_result.unexpected_keys)
        if missing:
            print(f"[WARN] Missing keys when loading checkpoint: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys in checkpoint: {unexpected}")

    optimizer_loaded = False
    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer_loaded = True
        except Exception as exc:
            if rank == 0:
                print(f"[WARN] Failed to restore optimizer state: {exc}")
    elif optimizer is not None and rank == 0:
        print("[WARN] Optimizer state not found in checkpoint; continuing with fresh optimizer parameters.")

    scaler_loaded = False
    if loss_scaler is not None and isinstance(checkpoint, dict) and "scaler" in checkpoint:
        try:
            loss_scaler.load_state_dict(checkpoint["scaler"])
            scaler_loaded = True
        except Exception as exc:
            if rank == 0:
                print(f"[WARN] Failed to restore scaler state: {exc}")

    rng_loaded = False
    if isinstance(checkpoint, dict):
        try:
            rng_payload = checkpoint.get("rng_state_all_ranks", checkpoint.get("rng_state", None))
            rng_loaded = utils.restore_rng_state(rng_payload)
        except Exception as exc:
            if rank == 0:
                print(f"[WARN] Failed to restore RNG state: {exc}")

    if rank == 0:
        print(
            f"[INFO] Resume summary: optimizer_loaded={optimizer_loaded}, "
            f"scaler_loaded={scaler_loaded}, rng_loaded={rng_loaded}"
        )

    resume_epoch = checkpoint["epoch"] if isinstance(checkpoint, dict) and "epoch" in checkpoint else None
    best_loss = checkpoint["best_loss"] if isinstance(checkpoint, dict) and "best_loss" in checkpoint else None
    return resume_epoch, best_loss


def launch_tracking(terminal_args: argparse.Namespace) -> None:
    """Launch the training process for the tracking task."""
    args = prepare_tracking_args(machine=terminal_args.on)
    utils.apply_compile_override(args, getattr(terminal_args, "compile", "auto"))
    utils.resolve_args_paths(
        args,
        __file__,
        [
            "train_path",
            "test_path",
            "val_path",
            "output_dir",
            "log_dir",
            "init_ckpt",
            "resume",
            "resume_checkpoint",
        ],
    )

    # seed = args.seed
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # cudnn.benchmark = True

    # region --------------------------- distributed setup ---------------------------
    rank, local_rank, world_size, _, _ = utils.get_resources()
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        args.distributed = True
    else:
        args.distributed = False
    if args.device == "cuda":
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

    if args.log_dir and not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    utils.setup_for_distributed(rank == 0)
    # endregion ------------------------ distributed setup ------------------------

    utils.print_perf_config(args, extra_keys=["disable_scheduler"])
    utils.configure_cuda_runtime(args, device, rank)
    utils.configure_inductor_cudagraphs(args, rank)

    if args.auto_resume and not getattr(args, "resume_checkpoint", "") and not getattr(args, "resume", ""):
        output_dir = Path(args.output_dir)
        resume_candidates = [
            output_dir / "last_checkpoint-tracking.pth",
            output_dir / "checkpoint-tracking-best.pth",
        ]
        latest_generic = utils.find_latest_numbered_checkpoint(args.output_dir, "checkpoint-*.pth")
        if latest_generic:
            resume_candidates.append(Path(latest_generic))
        for candidate in resume_candidates:
            if candidate and Path(candidate).exists():
                args.resume_checkpoint = str(candidate)
                if args.rank == 0:
                    print(f"[INFO] Auto-resume checkpoint detected: {args.resume_checkpoint}")
                break

    # ------------------------------- datasets via DataManager -------------------------------
    train_m = DataManager(is_train=True, args=args, type_t='supervised', world_size=world_size, rank=rank)
    test_m = DataManager(is_train=False, args=args, type_t='supervised', world_size=world_size, rank=rank)
    val_m = None
    if getattr(args, 'val_path', None):
        val_m = DataManager(is_train=False, args=args, type_t='supervised', world_size=world_size, rank=rank, specify_data_path=args.val_path)

    train_loader = train_m.get_tracking_dataloader(args)
    test_loader = test_m.get_tracking_dataloader(args)
    val_loader = val_m.get_tracking_dataloader(args) if val_m is not None else None
    _summarize_tracking_dataset("train", train_m.dataset, train_loader, args.rank)
    _summarize_tracking_dataset("test", test_m.dataset, test_loader, args.rank)
    if val_m is not None:
        _summarize_tracking_dataset("val", val_m.dataset, val_loader, args.rank)

    if getattr(terminal_args, "dry_run", False):
        if args.rank == 0:
            print("[DRY-RUN] DataLoader creation completed. Exiting before model/training.")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return

    # region ------------------------------- model ---------------------------------
    #model_kwargs = args.__dict__.copy()
    #for k in ["model", "init_ckpt", "nb_classes"]:
    #    model_kwargs.pop(k, None)
    model = create_tracking_model(
        args.model,
        **args.__dict__
    )
    model = utils.maybe_compile_model(model, args)

    # Informative print: confirm regression head is active
    try:
        from models.tracking_model import RegressionHead  # local import for isinstance check
        head_is_reg = isinstance(model.head, RegressionHead)
    except Exception:
        head_is_reg = False
    print(f"[INFO] Model built: {args.model} | Head: {type(model.head).__name__} | Regression active: {head_is_reg}")

    model.to(device)
    model_without_ddp = model
    model, model_without_ddp = utils.wrap_model_distributed(model, args, rank)

    # endregion ------------------------------ model ---------------------------------

    criterion = nn.MSELoss()
    optimizer = create_optimizer(args, model_without_ddp)
    autocast_dtype = utils.get_autocast_dtype(getattr(args, "amp_dtype", "fp32"))
    use_fp16_scaler = device.type == "cuda" and autocast_dtype == torch.float16
    loss_scaler = utils.NativeScalerWithGradNormCount() if use_fp16_scaler else None
    if args.rank == 0:
        print(
            f"[INFO] Tracking AMP config: amp_dtype={getattr(args, 'amp_dtype', 'fp32')}, "
            f"autocast_dtype={autocast_dtype}, scaler_enabled={loss_scaler is not None}"
        )

    best_loss = float("inf")
    if getattr(args, "auto_resume", False):
        resume_path = getattr(args, "resume_checkpoint", "") or getattr(args, "resume", "")
        if resume_path:
            try:
                resume_epoch, resume_best = _resume_from_checkpoint(
                    model=model_without_ddp,
                    optimizer=optimizer,
                    checkpoint_path=resume_path,
                    loss_scaler=loss_scaler,
                    rank=args.rank,
                )
                if resume_epoch is not None:
                    args.start_epoch = max(args.start_epoch, resume_epoch + 1)
                    if args.rank == 0:
                        print(f"[INFO] Training will resume from epoch {args.start_epoch}")
                if resume_best is not None:
                    best_loss = resume_best
                    if args.rank == 0:
                        print(f"[INFO] Best validation loss restored from checkpoint: {best_loss:.4f}")
            except FileNotFoundError:
                if args.rank == 0:
                    print(f"[WARN] Resume checkpoint not found: {resume_path}")
            except Exception as exc:
                if args.rank == 0:
                    print(f"[WARN] Failed to resume from checkpoint '{resume_path}': {exc}")
        elif args.rank == 0:
            print("[INFO] auto_resume is enabled but 'resume_checkpoint' path is empty. Skipping resume.")

    if args.rank == 0 and args.epochs <= args.start_epoch:
        print(f"[WARN] Training config has epochs={args.epochs} and start_epoch={args.start_epoch}; loop will exit immediately.")

    total_batch_size = args.batch_size * world_size
    num_training_steps_per_epoch = len(train_loader)
    if num_training_steps_per_epoch <= 0:
        raise ValueError(
            "num_training_steps_per_epoch=0: dataset troppo piccolo rispetto al batch effettivo. "
            f"dataset_len={train_m.dataset_len}, total_batch_size={total_batch_size}, "
            f"batch_size={args.batch_size}, world_size={world_size}."
        )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    lr_schedule_values = None
    wd_schedule_values = None
    if getattr(args, "disable_scheduler", False):
        print("[INFO] Learning-rate scheduler disabled: optimizer will use fixed lr.")
    else:
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, start_warmup_value=args.warmup_lr, warmup_steps=args.warmup_steps)
        wd_schedule_values = utils.cosine_scheduler(
            args.weight_decay, args.weight_decay_end,
            args.epochs, num_training_steps_per_epoch
        )
        print(
            f"[INFO] Scheduler enabled: steps_per_epoch={num_training_steps_per_epoch}, "
            f"lr_start={float(lr_schedule_values[0]):.8g}, lr_end={float(lr_schedule_values[-1]):.8g}, "
            f"wd_max={float(max(wd_schedule_values)):.8g}, wd_min={float(min(wd_schedule_values)):.8g}"
        )



    start_time = time.time()
    last_epoch_ran = args.start_epoch - 1
    for epoch in range(args.start_epoch, args.epochs):
        print(f"start epoch{epoch}")
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            amp_dtype=getattr(args, "amp_dtype", "fp32"),
            compute_geo_metric=False,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
        )
        print("train step")
        last_epoch_ran = epoch
        # Evaluate on test and optionally validation
        test_stats = evaluate(model, criterion, test_loader, device, amp_dtype=getattr(args, "amp_dtype", "fp32"))
        val_stats = evaluate(model, criterion, val_loader, device, amp_dtype=getattr(args, "amp_dtype", "fp32")) if val_loader is not None else None
        print("eval step")
        test_loss = test_stats.get("loss", float("inf"))
        val_loss = val_stats.get("loss", float("inf")) if val_stats is not None else float("inf")
        monitor_loss = min(test_loss, val_loss)
        is_new_best = monitor_loss < best_loss and epoch > args.start_epoch_for_saving_best_ckpt
        best_rng_state_payload = None
        if args.output_dir and is_new_best:
            local_rng_state = utils.capture_rng_state()
            best_rng_state_payload = utils.gather_rng_state_all_ranks(local_rng_state)
        if args.output_dir and is_new_best and args.rank == 0:
            best_loss = monitor_loss
            checkpoint_path = os.path.join(args.output_dir, "checkpoint-tracking-best.pth")
            model_state = model_without_ddp.state_dict()
            checkpoint_payload = {
                "model": model_state,
                "state_dict": model_state,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
                "args": args.__dict__,
            }
            if loss_scaler is not None:
                checkpoint_payload["scaler"] = loss_scaler.state_dict()
            if isinstance(best_rng_state_payload, list):
                checkpoint_payload["rng_state_all_ranks"] = best_rng_state_payload
            elif best_rng_state_payload is not None:
                checkpoint_payload["rng_state"] = best_rng_state_payload
            torch.save(
                checkpoint_payload,
                checkpoint_path,
            )
            print(f"[INFO] Best checkpoint saved at {checkpoint_path}")
        if is_new_best:
            best_loss = monitor_loss

        log_stats = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
        }
        if val_stats is not None:
            log_stats.update({f"val_{k}": v for k, v in val_stats.items()})

        if args.output_dir and args.rank == 0:
            log_path = os.path.join(args.output_dir, "log.txt")
            with open(log_path, "a") as f:
                f.write(
                    json.dumps(
                        {k: _format_log_value(k, v) for k, v in log_stats.items()}
                    )
                    + "\n"
                )


    total_time = time.time() - start_time
    total_time_str = str(time.strftime('%H:%M:%S', time.gmtime(total_time)))
    print(f"Training time {total_time_str}")

    # ultimo salvataggio per riprendere da dove abbiamo lasciato
    final_rng_state_payload = None
    if args.output_dir:
        local_rng_state = utils.capture_rng_state()
        final_rng_state_payload = utils.gather_rng_state_all_ranks(local_rng_state)
    if args.output_dir and args.rank == 0:
        last_checkpoint_path = os.path.join(args.output_dir, "last_checkpoint-tracking.pth")
        model_state = model_without_ddp.state_dict()
        checkpoint_payload = {
            "model": model_state,
            "state_dict": model_state,
            "optimizer": optimizer.state_dict(),
            "epoch": last_epoch_ran,
            "best_loss": best_loss,
            "args": args.__dict__,
        }
        if loss_scaler is not None:
            checkpoint_payload["scaler"] = loss_scaler.state_dict()
        if isinstance(final_rng_state_payload, list):
            checkpoint_payload["rng_state_all_ranks"] = final_rng_state_payload
        elif final_rng_state_payload is not None:
            checkpoint_payload["rng_state"] = final_rng_state_payload
        torch.save(
            checkpoint_payload,
            last_checkpoint_path,
        )
        print(f"[INFO] Last checkpoint saved at {last_checkpoint_path} (epoch={last_epoch_ran})")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Cyclone centre tracking")
    parser.add_argument('--on',
        type=str,
        default='leonardo',
        #metavar='NAME',
        help='[ewc, leonardo]'
    ),
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Build tracking dataloaders, print dataset counts, and exit.",
    )
    parser.add_argument(
        '--compile',
        type=str,
        default='auto',
        choices=['auto', 'on', 'off'],
        help='Override compile_model: auto=usa arguments.py, on=True, off=False'
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    launch_tracking(parsed)
