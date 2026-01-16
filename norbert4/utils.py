import torch
import torch.distributed as dist
import os
import random
import math


def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        lr = max(min_factor, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
        return lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def cosine_schedule_with_warmup_cooldown(optimizer, num_warmup_steps: int, num_cooldown_steps: int, num_training_steps: int, min_factor: float):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step >= num_training_steps - num_cooldown_steps:
            return min_factor * float(num_training_steps - current_step) / float(max(1, num_cooldown_steps))

        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        lr = max(min_factor, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
        return lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def mask_schedule(min_mask_p: float, max_mask_p: float, num_warmup_steps: int, num_cooldown_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return min_mask_p + (max_mask_p - min_mask_p) * float(current_step) / float(max(1, num_warmup_steps))
    elif current_step >= num_training_steps - num_cooldown_steps:
        return min_mask_p + (max_mask_p - min_mask_p) * float(num_training_steps - current_step) / float(max(1, num_cooldown_steps))

    return max_mask_p


class MaskScheduler:
    def __init__(self, min_mask_p: float, max_mask_p: float, num_warmup_steps: int, num_cooldown_steps: int, num_training_steps: int):
        self.min_mask_p = min_mask_p
        self.max_mask_p = max_mask_p
        self.num_warmup_steps = num_warmup_steps
        self.num_cooldown_steps = num_cooldown_steps
        self.num_training_steps = num_training_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.num_warmup_steps:
            return self.min_mask_p + (self.max_mask_p - self.min_mask_p) * float(self.current_step) / float(max(1, self.num_warmup_steps))
        elif self.current_step >= self.num_training_steps - self.num_cooldown_steps:
            return self.min_mask_p + (self.max_mask_p - self.min_mask_p) * float(self.num_training_steps - self.current_step) / float(max(1, self.num_cooldown_steps))

        return self.max_mask_p

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
    
    def state_dict(self):
        return {
            'current_step': self.current_step
        }

def trapezoid_schedule(optimizer, num_warmup_steps: int, num_cooldown_steps: int, num_training_steps: int):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step >= num_training_steps - num_cooldown_steps:
            return float(num_training_steps - current_step) / float(max(1, num_cooldown_steps))

        lr = 1.0
        return lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def flat_with_warmup_schedule(optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        lr = 1.0
        return lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def trapezoid_schedule_sqrt(optimizer, num_warmup_steps: int, num_cooldown_steps: int, num_training_steps: int):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step >= num_training_steps - num_cooldown_steps:
            return 1 - math.sqrt((current_step - (num_training_steps - num_cooldown_steps)) / max(1, num_cooldown_steps))

        lr = 1.0
        return lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class CosineWDSchedule(object):
    def __init__(self, optimizer, ref_wd, T_max, final_wd=0.0, step=0):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = step

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if group["weight_decay"] != 0.0:
                group['weight_decay'] = new_wd
        return new_wd


def seed_everything(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0