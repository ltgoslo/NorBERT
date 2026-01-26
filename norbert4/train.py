import os
import sys
import argparse
from tqdm import tqdm
from socket import gethostname
import json
import math
from pathlib import Path
from functools import partial
from contextlib import nullcontext
import datetime
from glob import glob
from statistics import mean

from tokenizers import Tokenizer
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.attention.flex_attention import create_block_mask
import torch._dynamo
from torch.nn.parallel import DistributedDataParallel as DDP

from model import Model
from muon import Muon
from stable_lamb import StableLamb
from utils import trapezoid_schedule, trapezoid_schedule_sqrt, is_main_process, seed_everything
from dataset import MaskedDataset, CausalDataset
from validation_dataset import ValidationDataset

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.suppress_errors = True

if int(os.environ["SLURM_PROCID"]) == 0:
    import wandb


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", required=True, nargs="+", type=Path, help="List of train dataset names.")
    parser.add_argument("--dataset_weights", required=True, nargs="+", type=float, help="Weights of each train datasets during training.")
    parser.add_argument("--name", default="hybrid_100M_Base_Split", type=str, help="Name of the run.")
    parser.add_argument("--wandb_project", default="gpt-bert", type=str, help="Name of the WandB project to log into.")
    parser.add_argument("--wandb_entity", default="ltg", type=str, help="The entity to log to on WandB (typically your wandb username).")
    parser.add_argument("--config_file", default="./config/base.json", type=Path, help="The BERT model config")
    parser.add_argument("--tokenizer_path", default="../tokenizer.json", type=Path, help="Path to the tokenizer.")
    parser.add_argument("--output_dir", default="../model_checkpoints", type=Path, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--checkpoint_foldername", default=None, type=Path, help="The checkpoint filename to resume training.")
    parser.add_argument("--hybrid_numerator", default=1, type=int, help="The numerator of the hybrid ratio.")
    parser.add_argument("--hybrid_denominator", default=2, type=int, help="The denominator of the hybrid ratio (the number of GPUs should be divisible by this number).")
    parser.add_argument("--max_seq_length", default=8192*2, type=int, help="Sequence length for training.")
    parser.add_argument("--window_length", default=512, type=int, help="Window length for training.")
    parser.add_argument("--local_batch_size", default=2, type=int, help="Batch size for training per GPU.")
    parser.add_argument("--global_batch_size", default=512, type=int, help="Total batch size for training per GPUs and per grad accumulation step.")
    parser.add_argument("--learning_rate", default=1e-2, type=float, help="The initial learning rate for Muon.")
    parser.add_argument("--embed_lr", default=1e-2, type=float, help="The initial learning rate for Embedding parameters.")
    parser.add_argument("--scalar_lr", default=1e-2, type=float, help="The initial learning rate for Scalar parameters.")
    parser.add_argument("--head_lr", default=0.1/1024**0.5, type=float, help="The initial learning rate for Scalar parameters.")
    parser.add_argument("--number_of_tokens", default=6e11, type=int, help="Total number of tokens to train on.")
    parser.add_argument("--max_steps", default=None, type=int)
    parser.add_argument("--document_skip", default=1, type=int)
    parser.add_argument("--validate_every", default=1_000, type=int, help="Run validation after every X training shards.")
    parser.add_argument("--validation_steps", default=1, type=int, help="Number of validation steps.")
    parser.add_argument("--validation_path", default="/cluster/work/projects/nn9851k/mariiaf/hplt/deu_Latn/tokenized_shards/validation.pt.gz")
    parser.add_argument("--log_stats_every", default=100, type=int, help="Log stats every X steps.")
    parser.add_argument("--scheduler", default="trapezoid", type=str, help="Which learning rate scheduler to use.", choices=["trapezoid", "cosine"])
    parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--cooldown_proportion", default=0.16, type=float, help="Proportion of training to perform linear learning rate cooldown for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--save_every', type=int, default=1_000, help="save every X steps")
    parser.add_argument('--checkpoint_every', type=int, default=10_000, help="create a model chekpoint every X steps")
    parser.add_argument("--mask_p_start", default=0.3, type=float, help="Initial masking probability.")
    parser.add_argument("--mask_p_end", default=0.15, type=float, help="Final masking probability.")
    parser.add_argument("--mask_random_p", default=0.1, type=float, help="Probability of replacing the masked token with a random token.")
    parser.add_argument("--mask_keep_p", default=0.1, type=float, help="Probability of keeping the masked token.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--optimizer_eps", default=1e-8, type=float, help="Optimizer epsilon.")
    parser.add_argument("--optimizer_beta1", default=0.9, type=float, help="Optimizer beta1.")
    parser.add_argument("--optimizer_beta2", default=0.95, type=float, help="Optimizer beta2.")
    parser.add_argument("--max_gradient", default=1e9, type=float, help="Max value for gradient clipping.")
    parser.add_argument('--n_special_tokens', default=16, type=int, help="Number of special tokens.")
    parser.add_argument('--z_loss_weight', default=0.0, type=float, help="Weight for the z loss.")
    parser.add_argument("--experiment", default="ablations", type=str)
    parser.add_argument("--optimizer", default="muon", type=str, choices=["muon", "lamb"])
    parser.add_argument("--untie", default=False, action="store_true")
    parser.add_argument("--momentum", default=0.95, type=float)
    parser.add_argument("--wd_scales", default=False, action="store_true")
    parser.add_argument("--mask_token", default="[MASK]")
    parser.add_argument("--cls_token", default="[CLS]")
    parser.add_argument("--pad_token", default="[PAD]")
    parser.add_argument("--train_format", default="pt.gz")
    parser.add_argument("--window_update", default="2,4,8,16,16")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_path = (args.output_dir / args.name)
    args.output_path.mkdir(parents=True, exist_ok=True)

    return args


def setup_training(args, tokenizer):
    assert torch.cuda.is_available()
    args.n_gpu = torch.cuda.device_count()
    args.tokens_per_batch = args.global_batch_size * args.max_seq_length
    args.window_update = tuple([int(number) for number in args.window_update.split(",")])
    if args.max_steps is None:
        args.max_steps = (args.number_of_tokens // args.tokens_per_batch) + 1
    else:
        args.number_of_tokens = args.max_steps * args.tokens_per_batch

    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["SLURM_PROCID"])
    args.gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert args.gpus_per_node == torch.cuda.device_count()  # Might create errors on ROCm
    print(f"Hello from rank {args.rank} of {args.world_size} on {gethostname()} where there are {args.gpus_per_node} allocated GPUs per node.", flush=True)
    
    args.accumulate_steps = max(1, (args.global_batch_size // args.world_size) // args.local_batch_size)

    assert args.world_size % args.hybrid_denominator == 0
    if args.rank * args.hybrid_denominator < args.hybrid_numerator * args.world_size:
        args.dataset_type = "masked"
    else:
        args.dataset_type = "causal"
    print(f"Dataset type: {args.dataset_type}", flush=True)

    args.local_rank = args.rank % args.gpus_per_node

    dist.init_process_group(
        backend="nccl",
        init_method='env://', # default
        rank=args.rank,
        world_size=args.world_size,
        timeout=datetime.timedelta(minutes=60)
    )

    seed_everything(args.seed + args.rank)

    args.shard_rank = args.rank % args.number_of_shards
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    print(f"RCCL started on device {args.device}", flush=True)
    print(f"host: {gethostname()}, rank: {args.rank}, local_rank: {args.local_rank}, shard_rank: {args.shard_rank}")

    args.vocab_size = tokenizer.get_vocab_size()

    if is_main_process():
        wandb.init(
            name=args.name,
            project=args.wandb_project,
            entity=args.wandb_entity
        )
        wandb.config.update(args)
        wandb.save(sys.argv[0], policy="now")


def load_config(args):
    with args.config_file.open("r") as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)
    return args


def prepare_model_and_optimizer(args):
    args = load_config(args)
    model = Model(args)

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update(args, allow_val_change=True)
        wandb.config.update({"n_params": n_params}, allow_val_change=True)
        print(model)
        print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    for p in model.parameters():
        if not p.data.is_contiguous():
            print("Warning: non-contiguous parameter data", flush=True)
            p.data = p.data.contiguous()       
    
    model.cuda(args.device)

    # NanoGPT does this before initializing the optimizer and scheduler
    model = torch.compile(model, dynamic=False)

    ddp_model = DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        # bucket_cap_mb=torch.cuda.get_device_properties(args.device).total_memory,
        broadcast_buffers=False,
        gradient_as_bucket_view=False,
        # static_graph=True
    )
    print(f"Model initialized on device {args.device}", flush=True)

    hidden_matrix_params = [(n, p) for n, p in model.encoder.named_parameters() if p.ndim == 2 and ("q_scale" not in n and "k_scale" not in n)]
    if hasattr(model.classifier, "projection"):
        hidden_matrix_params.append(("classifier.projection.weight", model.classifier.projection.weight))
    embed_params = [(n, p) for n, p in model.named_parameters() if "embedding" in n and "norm" not in n and "scale" not in n]
    scalar_params = [(n, p) for n, p in model.named_parameters() if (p.ndim < 2 and ("scale" in n or "gamma" in n or "weight" in n)) or "q_scale" in n or "k_scale" in n]
    bias_params = [(n, p) for n, p in model.named_parameters() if p.ndim < 2 and ("scale" not in n and "gamma" not in n and "weight" not in n)]

    muon_parameters = [p for _, p in hidden_matrix_params]
    optimizer_grouped_parameters = [
        {"params": [p for _, p in embed_params], "lr": args.embed_lr, "weight_decay": 0.0},
        {"params": [p for _, p in scalar_params], "lr": args.scalar_lr, "weight_decay": args.weight_decay if args.wd_scales else 0.0},
        {"params": [p for _, p in bias_params], "lr": args.scalar_lr, "weight_decay": 0.0}
    ]

    if is_main_process():
        print("Parameters with Muon Optimizer:")
        for n, _ in hidden_matrix_params:
            print(n)
        print(f"\nParameters with lr {args.embed_lr}:")
        for n, _ in embed_params:
            print(n)
        print(f"\nParameters with lr {args.scalar_lr}:")
        for n, _ in scalar_params:
            print(n)
        for n, _ in bias_params:
            print(n)
        print(flush=True)

    optimizer1 = torch.optim.AdamW(
        optimizer_grouped_parameters,
        args.learning_rate,
        betas=(args.optimizer_beta1, args.optimizer_beta2),
        eps=args.optimizer_eps,
        fused=True
    )
    if args.optimizer == "lamb":
        optimizer1 = StableLamb(
            optimizer_grouped_parameters,
            args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps
        )
        optimizer2 = StableLamb(
            muon_parameters,
            args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps
        )
    elif args.optimizer == "muon":
        optimizer1 = torch.optim.AdamW(
            optimizer_grouped_parameters,
            args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
            fused=True
        )
        optimizer2 = Muon(muon_parameters, lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum, rank=args.rank, world_size=args.world_size)
    
    optimizers = [optimizer1, optimizer2]

    schedulers = [
        trapezoid_schedule(
            optimizer,
            int(args.max_steps * args.warmup_proportion),
            int(args.max_steps * args.cooldown_proportion),
            args.max_steps
        ) for optimizer in optimizers
    ]

    global_step = 0
    if args.checkpoint_foldername is not None:
        path_to_checkpoint = args.checkpoint_foldername / "state_dict.bin"
        state_dict = torch.load(path_to_checkpoint, map_location=args.device)
        model.load_state_dict(state_dict["model"])
        for optimizer, state in zip(optimizers, state_dict["optimizers"]):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(schedulers, state_dict["schedulers"]):
            scheduler.load_state_dict(state)
        global_step = state_dict["global_step"]

    return model, ddp_model, optimizers, schedulers, global_step


def causal_mask_mode(max_seq_length, block_lengths, b, _, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    sliding_mask = (q_idx - kv_idx) < max_seq_length
    block_mask = block_lengths[b, q_idx] == block_lengths[b, kv_idx]
    return causal_mask & sliding_mask & block_mask


# We should try to cache this for speed apparently
def create_causal_mask(max_seq_length, block_lengths, args):
    length = torch.tensor(block_lengths.size(1), device="cuda")
    return create_block_mask(
        partial(causal_mask_mode, max_seq_length, block_lengths),
        block_lengths.size(0), 1, length, length, _compile=False
    )


def bidirectional_mask_mode(max_seq_length, block_lengths, b, _, q_idx, kv_idx):
    sliding_mask = ((q_idx - kv_idx) < max_seq_length) & ((kv_idx - q_idx) < max_seq_length)
    block_mask = block_lengths[b, q_idx] == block_lengths[b, kv_idx]
    return sliding_mask & block_mask


def create_bidirectional_mask(max_seq_length, block_lengths, args):
    length = torch.tensor(block_lengths.size(1), device="cuda")
    return create_block_mask(
        partial(bidirectional_mask_mode, max_seq_length, block_lengths),
        block_lengths.size(0), 1, length, length, _compile=False
    )


def old_causal_mask_mode(sequence_lengths, b, _, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    document_mask = sequence_lengths[q_idx] == sequence_lengths[kv_idx]
    return causal_mask & document_mask


def old_bidirectional_mask_mode(sequence_lengths, b, _, q_idx, kv_idx):
    document_mask = sequence_lengths[q_idx] == sequence_lengths[kv_idx]
    return document_mask


@torch.no_grad()
def update_window_length(global_step, args, model):
    if (global_step + 1) / args.max_steps >= 0.9:
        window_length = args.max_seq_length // args.window_update[0]
    elif (global_step + 1) / args.max_steps >= 0.8:
        window_length = args.max_seq_length // args.window_update[1]
    elif (global_step + 1) / args.max_steps >= 0.7:
        window_length = args.max_seq_length // args.window_update[2]
    elif (global_step + 1) / args.max_steps >= 0.5:
        window_length = args.max_seq_length // args.window_update[3]
    else:
        window_length = args.max_seq_length // args.window_update[4]

    if window_length != args.window_length:
        args.window_length = window_length
        model.set_window_length(args.window_length)

    return model


@torch.no_grad()
def old_get_batch(args, dataset, global_step):
    dataset.set_global_step(global_step)
    batch = dataset.next(args.max_seq_length, 1)
    input_ids, target_ids, doc_ids, mask_p = [t.cuda(non_blocking=True) for t in batch]
    input_ids, target_ids = input_ids.t(), target_ids.t()
    mask_p = mask_p.mean()

    return input_ids, target_ids, doc_ids, mask_p



@torch.no_grad()
def get_batch(args, dataset, global_step):
    batch = dataset.next(args.local_batch_size)
    input_ids, target_ids, block_lengths, mask_p = [t.cuda(non_blocking=True) for t in batch]
    input_ids, target_ids = input_ids.t(), target_ids.t()
    mask_p = mask_p.mean()

    if global_step > 200:
        window_length = args.max_seq_length
    elif global_step > 100:
        window_length = args.max_seq_length // 4
    else:
        window_length = args.max_seq_length // 16

    if args.dataset_type == "masked":
        block_mask = create_bidirectional_mask(
            window_length, block_lengths, args
        )
    else:
        block_mask = create_causal_mask(
            window_length, block_lengths, args
        )

    return input_ids, block_mask, target_ids, mask_p, window_length


@torch.no_grad()
def validation_loop(ddp_model, valid_dataset, args, global_step):
    ddp_model = ddp_model.eval()

    # Initialize the progress bar
    progress_bar = tqdm(total=args.validation_steps, initial=global_step, disable=not is_main_process(), desc="Validation iteration")

    losses, accuracies = [], []
    for step in range(args.validation_steps):
        next_batch = old_get_batch(args, valid_dataset, global_step)
        input_ids, target_ids, sequence_lengths, mask_p = next_batch 
        output = ddp_model(input_ids, sequence_lengths, target_ids)
        loss, accuracy = output.loss, output.accuracy

        # accumulate the metrics across GPUs
        metrics = torch.stack([loss, accuracy])
        dist.all_reduce(metrics, dist.ReduceOp.AVG)
        loss, accuracy = metrics.tolist()
        losses.append(loss)
        accuracies.append(accuracy)
        progress_bar.update()
    # log the metrics
    if is_main_process():
        wandb.log(
            {
                "val/loss": mean(losses),
                "val/accuracy": mean(accuracies) * 100.0,
            },
            step=global_step
        )
    progress_bar.close()


def training_loop(model, ddp_model, train_dataset, valid_dataset, optimizers, schedulers, global_step, args):
    model = model.train()
    model.zero_grad(set_to_none=True)

    # initialize the metrics
    total_loss, total_accuracy, total_z_loss, total_mask_p, total_grad_norm = 0.0, 0.0, 0.0, 0.0, 0.0

    # calculate the number of forward passes to perform
    num_steps = int(args.max_steps * args.accumulate_steps)

    # Initialize the progress bar
    progress_bar = tqdm(total=args.max_steps, initial=global_step, disable=not is_main_process(), desc="Train iteration")

    # get the first batch
    model = update_window_length(global_step, args, model)

    # iterate over the steps
    for local_step in range(num_steps):
        next_batch = old_get_batch(args, train_dataset, global_step)

        input_ids, target_ids, doc_ids, mask_p = next_batch # doc_ids are sequence_lengths

        # forward pass, do a more detailed check of the model every 100 steps
        # with ModelLogger(enable=global_step % 100 == 0, module=model):
        with ddp_model.no_sync() if (local_step + 1) % args.accumulate_steps != 0 else nullcontext():

            
            output = ddp_model(input_ids, doc_ids, target_ids)

            loss, accuracy, z_loss, num_tokens = output.loss, output.accuracy, output.z_loss, output.num_tokens
            
            # calculate the weight for the loss (either token-weighted or not)
            #total_tokens = torch.tensor(num_tokens, device=args.device, dtype=torch.long)
            #torch.distributed.all_reduce(total_tokens, torch.distributed.ReduceOp.SUM)
            
            #weight = args.world_size * num_tokens / total_tokens / args.accumulate_steps  # TODO: Are we happy on how we assign weight of each objective?
            weight = 1.0 / args.accumulate_steps
            if mask_p != 0:
                weight = weight * (mask_p / args.mask_p_start)

            # backward pass through both losses
            ((loss + args.z_loss_weight * z_loss) * weight).backward()

        # add the tracked metrics (for gradient accumulation)
        with torch.no_grad():
            total_loss += loss.detach() / args.accumulate_steps
            total_accuracy += accuracy / args.accumulate_steps
            total_z_loss += z_loss.detach() / args.accumulate_steps
            total_mask_p += mask_p  / args.accumulate_steps

        # gradient accumulation -- if we have accumulated enough gradients, we can perform the optimizer step; otherwise, we just continue and backpropagate through the next batch
        if (local_step + 1) % args.accumulate_steps != 0:
            continue

        # clip the gradients
        total_grad_norm += nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient) * weight
        
        # Muon Warmup
        frac = min(global_step / 500, 1)
        for group in optimizers[1].param_groups:
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

        # optimizer step
        for optimizer, scheduler in zip(optimizers, schedulers):
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            # be careful here, not all GPUs work with the same training objective
            if args.dataset_type == "masked":
                total_mlm_loss = total_loss / (args.hybrid_numerator / args.hybrid_denominator)
                total_mlm_accuracy = total_accuracy  / (args.hybrid_numerator / args.hybrid_denominator)
                total_clm_loss = torch.zeros_like(total_mlm_loss)
                total_clm_accuracy = torch.zeros_like(total_mlm_accuracy)
                total_mask_p = total_mask_p / (args.hybrid_numerator / args.hybrid_denominator)
            else:
                total_clm_loss = total_loss / (1 - args.hybrid_numerator / args.hybrid_denominator)
                total_clm_accuracy = total_accuracy / (1 - args.hybrid_numerator / args.hybrid_denominator)
                total_mlm_loss = torch.zeros_like(total_clm_loss)
                total_mlm_accuracy = torch.zeros_like(total_clm_accuracy)
                total_mask_p = torch.zeros_like(total_mask_p)

            # accumulate the metrics across GPUs
            metrics = torch.stack([total_loss, total_accuracy, total_z_loss, total_mask_p, total_mlm_loss, total_mlm_accuracy, total_clm_loss, total_clm_accuracy])
            dist.all_reduce(metrics, dist.ReduceOp.AVG)
            total_loss, total_accuracy, total_z_loss, total_mask_p, total_mlm_loss, total_mlm_accuracy, total_clm_loss, total_clm_accuracy = metrics.tolist()

            # log the metrics
            if is_main_process():
                wandb.log(
                    {
                        "train/loss": total_loss,
                        "train/z_loss": total_z_loss,
                        "train/perplexity": math.exp(total_loss),
                        "train/accuracy": total_accuracy * 100.0,
                        "train/mlm_loss": total_mlm_loss,
                        "train/mlm_accuracy": total_mlm_accuracy * 100.0,
                        "train/clm_loss": total_clm_loss,
                        "train/clm_accuracy": total_clm_accuracy * 100.0,
                        "stats/learning_rate": optimizer.param_groups[0]['lr'],
                        "stats/grad_norm": total_grad_norm,
                        "stats/window_length": args.window_length,
                        "stats/global_batch_size": args.global_batch_size * args.max_seq_length,
                        "stats/local_batch_size": args.local_batch_size * args.max_seq_length,  # Is this correct?
                        "stats/accumulate_steps": args.accumulate_steps,
                        "stats/mask_p": total_mask_p,
                        "global_step": global_step,

                    },
                    step=global_step
                )

        # zero the accumulated gradients and the metrics
        model.zero_grad(set_to_none=True)
        total_loss, total_accuracy, total_z_loss, total_mask_p, total_grad_norm = 0.0, 0.0, 0.0, 0.0, 0.0

        # save a backup of the model and the full training state
        if args.save_every:
            if global_step % args.save_every == 0:
                save(model, optimizers, schedulers, global_step, train_dataset, args)

        # save a checkpoint of the model and full training state
        if global_step % args.checkpoint_every == 0:
            save_checkpoint(model, optimizers, schedulers, global_step, train_dataset, args)

        if global_step % args.validate_every == 0:
            validation_loop(ddp_model, valid_dataset, args, global_step)

        # Exiting the training due to hitting max steps
        if global_step >= args.max_steps:
            progress_bar.close()
            return

        model = update_window_length(global_step, args, model)
        global_step += 1
        progress_bar.update()

    progress_bar.close()


def save(model, optimizers, schedulers, global_step, train_dataset, args):
    path_to_save_folder = args.output_path / "final"
    path_to_save_folder.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        torch.save(
            {
                "model": model.state_dict(),
                "optimizers": [optimizer.state_dict() for optimizer in optimizers],
                "schedulers": [scheduler.state_dict() for scheduler in schedulers],
                "global_step": global_step
            },
            path_to_save_folder / "state_dict.bin"
        )
    torch.save(
        train_dataset.get_state(),
        path_to_save_folder / f"dataset_info_{args.dataset_type}_{args.shard_rank}.bin"
    )


def save_checkpoint(model, optimizer, scheduler, global_step, train_dataset, args):
    path_to_save_folder = args.output_path / f"checkpoint_{global_step}"
    path_to_save_folder.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        torch.save(
            {
                "model": model.state_dict(),
                "optimizers": [optimizer.state_dict() for optimizer in optimizers],
                "schedulers": [scheduler.state_dict() for scheduler in schedulers],
                "global_step": global_step
            },
            path_to_save_folder / "state_dict.bin"
        )
    torch.save(
        train_dataset.get_state(),
        path_to_save_folder / f"dataset_info_{args.dataset_type}_{args.shard_rank}.bin"
    )


def load_train_dataset(args, tokenizer):
    if args.dataset_type == "masked":
        train_dataset = MaskedDataset(args.train_path, args.dataset_weights, tokenizer, args, args.max_seq_length, args.shard_rank)
    else:
        train_dataset = CausalDataset(args.train_path, args.dataset_weights, tokenizer, args, args.max_seq_length, args.shard_rank)

    if args.checkpoint_foldername is not None:
        dataset_state_dict = torch.load(args.checkpoint_foldername / f"dataset_info_{args.dataset_type}_{args.shard_rank}.bin", map_location="cpu")
        train_dataset.load_state(dataset_state_dict)

    return train_dataset


if __name__ == "__main__":
    args = parse_arguments()
    args.number_of_shards = sum([len(glob(str(train_path) + "*")) for train_path in args.train_path])
    if is_main_process():
        print(f"Total number of training shards: {args.number_of_shards}", flush=True)
    assert args.number_of_shards > 0
    assert args.number_of_shards <= int(os.getenv("SLURM_NTASKS"))
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    setup_training(args, tokenizer)
    model, ddp_model, optimizers, schedulers, global_step = prepare_model_and_optimizer(args)
    valid_dataset = ValidationDataset([args.validation_path], args.dataset_weights, tokenizer, args, args.max_seq_length, args.shard_rank)
    validation_loop(ddp_model, valid_dataset, args, global_step)
    train_dataset = load_train_dataset(args, tokenizer)
    training_loop(model, ddp_model, train_dataset, valid_dataset, optimizers, schedulers, global_step, args)

    save(model, optimizers, schedulers, args.max_steps, train_dataset, args)

