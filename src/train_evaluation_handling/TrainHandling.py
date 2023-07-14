import logging
import random
import statistics
import time

import numpy as np
import torch
from torch import nn, autocast
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .EvaluationHandling import save_evaluation_results
from .MetricCalculator import MetricCalculator
from ..data_handling.datasets.CountingDataset import CountingDataset
from ..util.PredictionVisualiser import PredictionVisualiser
from ..util.misc import accuracy, construct_confusion_matrix, mae_rmse, set_seed
from ..util.optimisation import get_scheduler
from ..util.plotting import plot_confusion_matrix

LOGGER = logging.getLogger()

DEFAULT_BATCH_SIZE = 128


def train_model(args, checkpoint_name, model, train_dataset, summary_writer, device, num_count_classes, criterion,
                clip_norm=0.0, eval_func=None, eval_freq=3, visualise_test_samples=None, batch_logging_freq=5,
                save_every_eval_epoch=False):
    LOGGER.info('Starting training')

    # training loop with validation after each epoch. Save the best model, and remember to use the lr scheduler.
    train_losses = []
    train_accs_1 = []
    train_mae = []
    train_rmse = []
    num_epochs = args.epochs

    # create dataloader
    if args.set_worker_seeds:
        train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, shuffle=True,
                                      batch_size=args.batch_size, pin_memory=True, drop_last=args.drop_last,
                                      worker_init_fn=lambda id: set_seed(args.seed*10+id))
    else:
        train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, shuffle=True,
                                      batch_size=args.batch_size, pin_memory=True, drop_last=args.drop_last)
    begin_second_stage = None
    second_stage_img_size = None
    end_epoch = num_epochs

    if args.use_second_stage:
        begin_second_stage = int((1 - args.second_stage_percentage) * num_epochs)
        if args.reset_optimisation:
            end_epoch = begin_second_stage - 1
        second_stage_img_size = args.second_img_size
        LOGGER.info(f'Setting image size to {second_stage_img_size} after {begin_second_stage} epochs')

    # prepare optimiser and scheduler
    lr = args.lr
    backbone_lr = args.backbone_lr
    if args.scale_lr:
        scaling_factor = args.batch_size / DEFAULT_BATCH_SIZE
        lr = scaling_factor * args.lr
        backbone_lr = scaling_factor * args.backbone_lr
        LOGGER.debug(
            f'Scaling learning rate by a factor of {scaling_factor}. Final learning rate: {lr}, backbone lr: {backbone_lr}')

    def create_optimizer_and_scheduler(start_epoch, end_epoch):
        optimizer = AdamW([
            {'params': model.backbone_parameters.parameters(), 'lr': backbone_lr},
            {'params': model.counting_parameters.parameters()}
        ], lr=lr, weight_decay=args.weight_decay)
        scheduler = get_scheduler(args, end_epoch+1, optimizer, start_epoch)
        return optimizer, scheduler

    optimizer, scheduler = create_optimizer_and_scheduler(0, end_epoch)

    num_epoch_digits = len(str(num_epochs))

    start = time.time()

    epoch_times = []

    if args.eval_before_training:
        epoch_range = range(-1, num_epochs)
    else:
        epoch_range = range(num_epochs)

    pred_visualiser = PredictionVisualiser([train_dataset[i] for i in range(2)], model, name='train', shot_num=3, device=device,
                                           denormalise=train_dataset.normalise_transform.reverse,
                                           density_scaling=args.density_scaling)
    test_pred_visualiser = None
    if visualise_test_samples is not None:
        test_pred_visualiser = []
        for name, samples in visualise_test_samples.items():
            cur_visualiser = PredictionVisualiser(samples, model, name=name, shot_num=3, device=device,
                                                        denormalise=train_dataset.normalise_transform.reverse,
                                                        density_scaling=args.density_scaling)
            test_pred_visualiser.append(cur_visualiser)

    if args.pytorch_amp and device.type != 'cuda':
        raise ValueError('PyTorch AMP is only supported on CUDA devices')
    if args.pytorch_amp and device.type == 'cuda':
        scaler = GradScaler()
        if clip_norm != 0.0:
            raise ValueError('Gradient clipping is not supported with PyTorch AMP')
    else:
        scaler = None

    def inference_and_loss(model_input, targets_dict, epoch):
        if epoch == -1:
            with torch.no_grad():
                raw_preds = model(*model_input)
                loss_dict = criterion(raw_preds, targets_dict)
        else:
            raw_preds = model(*model_input)
            loss_dict = criterion(raw_preds, targets_dict)
        return raw_preds, loss_dict

    batch_logging_loss = []

    for epoch in epoch_range:
        set_seed(args.seed + epoch)
        running_losses = 0
        running_train_acc_1 = 0
        running_mae = 0
        running_rmse = 0
        metric_calculator = MetricCalculator()
        num_samples = 0
        epoch_descr = f'Epoch {str(epoch + 1).zfill(num_epoch_digits)}/{num_epochs}'

        if begin_second_stage is not None and epoch == begin_second_stage:
            train_dataset.set_img_size(second_stage_img_size)
            if args.reset_optimisation:
                optimizer, scheduler = create_optimizer_and_scheduler(epoch, num_epochs)

        running_sub_losses_dict = {}

        if hasattr(train_dataset, 'set_difficulty_factor'):
            train_dataset.set_difficulty_factor(epoch, num_epochs)

        model.train()
        criterion.train()
        for batch_idx, (imgs, targets) in enumerate(tqdm(train_dataloader, leave=False, desc=epoch_descr, disable=args.disable_tqdm)):
            if scheduler is not None:
                scheduler.step(batch_idx/len(train_dataloader)+epoch)

            # training
            imgs = imgs.to(device, non_blocking=True)
            targets_dict = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

            min_shots = args.min_shots
            if args.use_constant_object_numbers:
                # if we are using a constant number of objects by default, the 0 shot setting is handled by the dataset
                min_shots = max(args.min_shots, 1)
            if args.vary_shots_per_sample:
                shot_num = torch.randint(min_shots, args.max_shots+1, (len(imgs),), device=device)
                if train_dataset.LABEL_DICT_IS_ZERO_SHOT in targets_dict:
                    zero_shot_mask = targets_dict[train_dataset.LABEL_DICT_IS_ZERO_SHOT]
                    shot_num[zero_shot_mask] = 0
                    if (zero_shot_mask.sum() > 0) and args.min_shots > 0:
                        raise ValueError('Cannot have zero-shot samples when min_shots > 0')
            else:
                shot_num = random.randint(min_shots, args.max_shots)
            model_input = (imgs, targets_dict[train_dataset.LABEL_DICT_BOXES], targets_dict[train_dataset.LABEL_DICT_REF_IMGS], shot_num)
            CountingDataset.modify_few_shot_target_dict(targets_dict, shot_num)

            if args.pytorch_amp:
                with autocast(device_type=device.type, dtype=torch.float16, enabled=args.pytorch_amp):
                    raw_preds, loss_dict = inference_and_loss(model_input, targets_dict, epoch)
            else:
                raw_preds, loss_dict = inference_and_loss(model_input, targets_dict, epoch)

            cls_preds = loss_dict['count_class_pred']
            cls_preds = cls_preds.clip(0, train_dataset.num_classes-1)

            if epoch >= 0:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss_dict['loss']).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_dict['loss'].backward()
                    if clip_norm != 0.0:
                        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    optimizer.step()
                batch_logging_loss.append(loss_dict['loss'].detach().cpu().item())

            # training statistics
            cur_batch_size = imgs.shape[0]
            # sub-losses
            for k, v in loss_dict.items():
                if k.startswith('loss_'):
                    if k in running_sub_losses_dict:
                        running_sub_losses_dict[k] += v * cur_batch_size
                    else:
                        running_sub_losses_dict[k] = v * cur_batch_size
            evaluation_labels = targets_dict[CountingDataset.LABEL_DICT_GLOBAL_COUNT]
            num_samples += cur_batch_size
            running_losses += loss_dict['loss'].item()*cur_batch_size
            acc = accuracy(cls_preds, evaluation_labels)
            running_train_acc_1 += acc*cur_batch_size
            if 'count_scalar_pred' in loss_dict:
                mae, rmse = mae_rmse(loss_dict['count_scalar_pred'], evaluation_labels)
                running_mae += mae*cur_batch_size
                running_rmse += rmse*cur_batch_size
            else:
                running_mae -= cur_batch_size
                running_rmse -= cur_batch_size

            cur_confusion_matrix = construct_confusion_matrix(cls_preds, evaluation_labels, num_count_classes)
            metric_calculator.update(raw_preds, loss_dict, cur_confusion_matrix)

            global_step = epoch * len(train_dataloader) + batch_idx
            if epoch >= 0 and global_step % batch_logging_freq == 0:
                avg_logging_batch_loss = sum(batch_logging_loss) / len(batch_logging_loss)
                summary_writer.add_scalars('batch-loss', {f'train': avg_logging_batch_loss}, global_step)
                batch_logging_loss = []

            if args.debug:
                summary_writer.add_scalars('debug-batch-loss', {f'train_{epoch}': loss_dict['loss'] / cur_batch_size}, batch_idx)
                if 'count_scalar_pred' in loss_dict:
                    batch_preds = loss_dict['count_scalar_pred']
                    summary_writer.add_scalars('batch-preds-mean', {f'train_{epoch}': batch_preds.mean()}, batch_idx)
                    summary_writer.add_scalars('batch-preds-std', {f'train_{epoch}': batch_preds.std()}, batch_idx)

        for name, weight in model.named_parameters():
            try:
                if weight.requires_grad and weight.grad is not None:
                    summary_writer.add_histogram(name, weight, epoch)
                    summary_writer.add_histogram(f'{name}.grad', weight.grad, epoch)
            except ValueError as e:
                LOGGER.debug(f'An error occurred when creating a histogram for weight {name} in epoch {epoch}: {e}')

        confusion_matrix, precisions, recalls = metric_calculator.class_metrics.get_metrics()
        confusion_fig = plot_confusion_matrix(confusion_matrix.detach().cpu().numpy().astype(int))
        summary_writer.add_figure('confusion_matrix', confusion_fig, global_step=epoch)
        rel_confusion_matrix = confusion_matrix / confusion_matrix.sum(1, keepdims=True)
        rel_confusion_fig = plot_confusion_matrix(rel_confusion_matrix.detach().cpu().numpy().round(2), ign_diagonal=False)
        summary_writer.add_figure('rel_confusion_matrix', rel_confusion_fig, global_step=epoch)
        train_losses.append(running_losses/num_samples)
        train_accs_1.append(running_train_acc_1/num_samples)
        train_mae.append(running_mae/num_samples)
        train_rmse.append(running_rmse/num_samples)

        pred_visualiser.visualise_prediction(summary_writer, epoch)
        if test_pred_visualiser is not None:
            for visualiser in test_pred_visualiser:
                visualiser.visualise_prediction(summary_writer, epoch)

        epoch_times.append(time.time() - start)

        metric_calculator.log_in_summary_writer(summary_writer, epoch)

        for loss_name, loss_sum in running_sub_losses_dict.items():
            summary_writer.add_scalars(loss_name, {'train': loss_sum/num_samples}, epoch)

        # add metrics to the summary writer
        summary_writer.add_scalars('Loss', {'train': train_losses[-1]}, epoch)
        if args.log_classification_metrics:
            summary_writer.add_scalars('Accuracy', {'train': train_accs_1[-1]}, epoch)
            summary_writer.add_scalars('AvgPrecision', {'train': precisions.mean()}, epoch)
            summary_writer.add_scalars('Precision', {str(c): precisions[c] for c in range(precisions.shape[0])}, epoch)
            summary_writer.add_scalars('AvgRecall', {'train': recalls.mean()}, epoch)
            summary_writer.add_scalars('Recall', {str(c): recalls[c] for c in range(recalls.shape[0])}, epoch)
        summary_writer.add_scalars('Time', {'train': epoch_times[-1]}, epoch)
        summary_writer.add_scalars('LR', {str(i): optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))}, epoch)
        if hasattr(train_dataset, 'difficulty_factor'):
            summary_writer.add_scalars('difficulty_factor', {'train': train_dataset.difficulty_factor}, epoch)
        if 'count_scalar_pred' in loss_dict:
            summary_writer.add_scalars('MAE', {'train': train_mae[-1]}, epoch)
            summary_writer.add_scalars('RMSE', {'train': train_rmse[-1]}, epoch)

        # save the latest model
        state_dict = model.state_dict()
        torch.save(state_dict, checkpoint_name)

        if (epoch % eval_freq == 0 or epoch < 0) and save_every_eval_epoch:
            torch.save(state_dict, checkpoint_name.rsplit('.', 1)[0] + f'_epoch_{epoch}.' + checkpoint_name.rsplit('.')[-1])

        logging_msg = f'{epoch_descr}\tTrain loss: {round(float(train_losses[-1]), 3): <8} ' \
                      f'MAE: {round(float(train_mae[-1]), 1): <8}'
        if args.log_classification_metrics:
            logging_msg += f'Acc 1: {round(float(train_accs_1[-1]), 3): <8}'\
                      f'AvgPrec: {round(float(precisions.mean()), 3): <8}'\
                      f'AvgRec: {round(float(recalls.mean()), 3): <8}'
        logging_msg += f'{round(epoch_times[-1], 1):<5}'
        if eval_func is not None and (epoch % eval_freq == 0 or epoch < 0):
            old_logging_level = LOGGER.level
            LOGGER.setLevel(logging.WARNING)
            eval_results = eval_func(model)
            save_evaluation_results(eval_results, summary_writers=[summary_writer], epoch=epoch, log_classification_metrics=args.log_classification_metrics)
            LOGGER.setLevel(old_logging_level)
            if 'FSC147' in eval_results[0]:
                logging_msg += f'FSC147 test MAE: {round(float(eval_results[0]["FSC147"]["mae"]), 1): <8}'

        LOGGER.debug(logging_msg)
        start = time.time()

    LOGGER.debug(f'Median time per epoch: {statistics.median(epoch_times):.2f}s')

    # save losses and accuracies
    results = np.vstack([np.array(train_losses), torch.stack(train_accs_1).detach().cpu().numpy(), epoch_times])
    np.savetxt(f'{checkpoint_name.split(".", 1)[0]}_results.csv', results, '%f')

    state_dict = model.state_dict()
    torch.save(state_dict, checkpoint_name)

    return model
