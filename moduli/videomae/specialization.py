import numpy as np
import os
import datetime
import time
import argparse
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
from optim_factory import create_optimizer
#from dataset import build_pretraining_dataset
from utils import get_model
from engine_for_pretraining import train_one_epoch, test
from arguments import prepare_args, Args  # NON TOGLIERE: serve a torch.load per caricare il mio modello addestrato
from dataset.data_manager import DataManager


utils.suppress_transformers_pytree_warning()



def launch_specialization_training(terminal_args):
    args = prepare_args(machine=terminal_args.on)
    # Backward compatibility: historical pretraining config used `data_path`
    # for train CSV, while DataManager expects `train_path`.
    if not getattr(args, "train_path", None):
        args.train_path = args.data_path
    if not getattr(args, "test_path", None):
        args.test_path = args.data_path
    

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

    # logging
    if args.log_dir and not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    setup_for_distributed(rank == 0)


    # LOAD MODEL
    pretrained_model = get_model(args)


    pretrained_model.to(device)  # rimane solo sul device generale o va messo nel local_rank?
    model_without_ddp = pretrained_model
    n_parameters = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
    #print("Model = %s" % str(model_without_ddp))
    #print('number of params: {} M'.format(n_parameters / 1e6))

    if args.distributed:
        pretrained_model = torch.nn.parallel.DistributedDataParallel(
            pretrained_model, device_ids=[args.gpu], output_device=args.gpu, 
            find_unused_parameters=False)
        model_without_ddp = pretrained_model.module



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


    optimizer = create_optimizer(args, model_without_ddp)
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
            normlize_target=args.normlize_target)

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
                        patch_size=patch_size[0], normlize_target=args.normlize_target, log_writer=log_writer)
            test_log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch}  #, 'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(test_log_stats) + "\n")

                # Save checkpoint only if current test loss improves over previous test loss.
                test_loss = test_stats.get("loss", None)
                if test_loss is not None:
                    improved = (prev_test_loss is None) or (float(test_loss) < float(prev_test_loss))
                    if improved:
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
    args = parser.parse_args()
    launch_specialization_training(args)
