import enum

from .CosineSchedulerWithWarmup import CosineSchedulerWithWarmup


class SupportedScheduler(enum.Enum):
    NO_SCHEDULER = 0
    Cosine_Scheduler = 1


def get_scheduler(args, epochs, optimizer, start_epoch=0):
    if args.scheduler is SupportedScheduler.NO_SCHEDULER:
        scheduler = None
    elif args.scheduler is SupportedScheduler.Cosine_Scheduler:
        scheduler = CosineSchedulerWithWarmup(optimizer=optimizer, warmup_percentage=args.warmup_percentage,
                                              epochs=epochs, min_lr=args.min_lr, start_epoch=start_epoch)
    else:
        raise ValueError(f'scheduler type {args.scheduler.name} unknown')
    return scheduler
