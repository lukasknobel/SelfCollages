"""
Based on https://github.com/Verg-Avesta/CounTR/blob/main/util/lr_sched.py
"""

import math


class CosineSchedulerWithWarmup:
    def __init__(self, optimizer, warmup_percentage, epochs, min_lr=0.0, start_epoch=0):

        # add epoch information
        self.start_epoch = start_epoch
        self.num_epochs = epochs - start_epoch
        self.warmup_epochs = self.num_epochs * warmup_percentage

        self.min_lr = min_lr
        self.optimizer = optimizer
        self.orig_lrs = [param_group["lr"] for param_group in self.optimizer.param_groups]

    def step(self, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < 0:
            # negative epochs are used for testing the model prior to any training and are therefore ignored
            return
        epoch = epoch - self.start_epoch
        if epoch < self.warmup_epochs:
            factor = epoch / self.warmup_epochs
            lrs = [lr * factor for lr in self.orig_lrs]
        else:
            factor = 0.5 * (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs)))
            lrs = [self.min_lr + (lr - self.min_lr) * factor for lr in self.orig_lrs]
        for lr, param_group in zip(lrs, self.optimizer.param_groups):
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr

    def __repr__(self):
        return f'{self.__class__.__name__}, warmup_epochs={self.warmup_epochs}, num_epochs={self.num_epochs}, ' \
               f'start_epoch={self.start_epoch}, min_lr={self.min_lr})'
