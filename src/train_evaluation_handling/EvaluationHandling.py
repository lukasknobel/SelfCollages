import logging
import os
import time

import numpy as np
import torch
from scipy.stats import kendalltau
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from .MetricCalculator import MetricCalculator
from ..data_handling.EvalDatasetFactory import create_eval_dataset
from ..data_handling.SupportedDatasets import SupportedEvalDatasets
from ..data_handling.datasets.CountingDataset import CountingDataset
from ..util import Constants
from ..util.Constants import RESULTS_FILE_NAME
from ..util.misc import accuracy, mae_rmse, construct_confusion_matrix, set_seed, prepare_dict_for_summary

LOGGER = logging.getLogger()


def evaluate_model(args, model, criterion, data_loader, num_count_classes, device, shot_num, return_preds=False):
    model.eval()
    if criterion is not None:
        criterion.eval()

    metric_calculator = MetricCalculator()
    running_losses = 0
    running_sub_losses_dict = {}
    running_acc_1 = 0
    running_mae = 0
    running_rmse = 0
    running_r2 = 0
    running_kendalltau = 0
    num_samples = 0
    if return_preds:
        preds_list = []
        targets_list = []
    with torch.no_grad():
        start = time.time()
        for imgs, targets in data_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

            model_input = (imgs, targets[Constants.TARGET_DICT_BOXES], targets[Constants.TARGET_DICT_REF_IMGS], shot_num)
            CountingDataset.modify_few_shot_target_dict(targets, shot_num)

            raw_preds = model(*model_input)
            if criterion is None:
                loss_dict = {'loss': torch.tensor(torch.nan)}
                cls_preds = raw_preds.round().to(torch.int64)
                scalar_preds = raw_preds
            else:
                loss_dict = criterion(raw_preds, targets)
                cls_preds = loss_dict['count_class_pred']
                cls_preds = cls_preds.clip(0, num_count_classes-1)
                scalar_preds = loss_dict['count_scalar_pred']
            loss = loss_dict['loss']
            if loss is None:
                loss = -1
            else:
                loss = loss.item()

            if return_preds:
                preds_list.append(raw_preds.detach().cpu())
                targets_list.append(targets[CountingDataset.LABEL_DICT_GLOBAL_COUNT].cpu())

            # train_evaluation_handling statistics
            cur_batch_size = imgs.shape[0]
            num_samples += cur_batch_size
            running_losses += loss * cur_batch_size
            acc = accuracy(cls_preds, targets[CountingDataset.LABEL_DICT_GLOBAL_COUNT])
            mae, rmse = mae_rmse(scalar_preds, targets[CountingDataset.LABEL_DICT_GLOBAL_COUNT])
            r2 = r2_score(targets[CountingDataset.LABEL_DICT_GLOBAL_COUNT].cpu().numpy(), scalar_preds.cpu().numpy())
            cur_kendalltau = kendalltau(targets[CountingDataset.LABEL_DICT_GLOBAL_COUNT].cpu().numpy(), scalar_preds.cpu().numpy()).correlation
            running_r2 += r2 * cur_batch_size
            running_kendalltau += cur_kendalltau * cur_batch_size
            running_acc_1 += acc * cur_batch_size
            running_mae += mae * cur_batch_size
            running_rmse += rmse * cur_batch_size
            cur_confusion_matrix = construct_confusion_matrix(cls_preds, targets[CountingDataset.LABEL_DICT_GLOBAL_COUNT], num_count_classes)
            metric_calculator.update(raw_preds, loss_dict, cur_confusion_matrix)

            # sub-losses
            for k, v in loss_dict.items():
                if k.startswith('loss_') and not k.rsplit('_', 1)[-1].isdigit():
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().item()
                    v = v * cur_batch_size
                    if k in running_sub_losses_dict:
                        running_sub_losses_dict[k] += v
                    else:
                        running_sub_losses_dict[k] = v

        eval_time = time.time() - start

        for loss_name, loss_sum in running_sub_losses_dict.items():
            running_sub_losses_dict[loss_name] = loss_sum/num_samples
    if return_preds:
        preds_list = np.concatenate(preds_list, axis=0)
        targets_list = np.concatenate(targets_list, axis=0)
        all_preds = np.stack([preds_list, targets_list])
        return running_losses / num_samples, running_sub_losses_dict, running_mae / num_samples, running_rmse / num_samples, \
               running_acc_1 / num_samples, running_r2 / num_samples, running_kendalltau / num_samples, metric_calculator, eval_time, \
               all_preds
    else:
        return running_losses / num_samples, running_sub_losses_dict, running_mae / num_samples, running_rmse / num_samples, \
               running_acc_1 / num_samples, running_r2 / num_samples, running_kendalltau / num_samples, metric_calculator, eval_time


def test_model(args, model, criterion, batch_size, test_dataset, num_count_classes, device, num_workers=0, seeds=None,
               return_last_preds=False):

    if seeds is None:
        seeds = [393]
    if isinstance(seeds, int):
        seeds = [seeds]

    sum_loss = 0
    sum_loss_dict = {}
    sum_acc = torch.tensor(0, dtype=torch.float, device=device)
    sum_r2 = 0
    sum_kendalltau = 0
    sum_mae = torch.tensor(0, dtype=torch.float, device=device)
    sum_rmse = torch.tensor(0, dtype=torch.float, device=device)
    sum_test_time = 0
    metrics_calculator = None
    cur_preds = None
    for seed in seeds:
        set_seed(seed)

        test_dataloader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=True)
        model.to(device)
        res = evaluate_model(args, model, criterion, test_dataloader, num_count_classes, device, test_dataset.num_shots, return_preds=return_last_preds)
        if return_last_preds:
            cur_loss, cur_loss_dict, cur_mae, cur_rmse, cur_acc, cur_r2, cur_kendalltau, cur_metrics_calculator, cur_test_time, cur_preds = res
        else:
            cur_loss, cur_loss_dict, cur_mae, cur_rmse, cur_acc, cur_r2, cur_kendalltau, cur_metrics_calculator, cur_test_time = res

        sum_loss += cur_loss
        if metrics_calculator is None:
            metrics_calculator = cur_metrics_calculator
        else:
            metrics_calculator.merge(cur_metrics_calculator)
        sum_acc += cur_acc
        sum_r2 += cur_r2
        sum_kendalltau += cur_kendalltau
        sum_mae += cur_mae
        sum_rmse += cur_rmse
        sum_test_time += cur_test_time

        for key, value in cur_loss_dict.items():
            sum_loss_dict[key] = sum_loss_dict.get(key, 0) + value

    num_seeds = len(seeds)
    avg_loss = sum_loss / num_seeds
    avg_loss_dict = {k: v / num_seeds for k, v in sum_loss_dict.items()}
    avg_acc = sum_acc / num_seeds
    avg_r2 = sum_r2 / num_seeds
    avg_kendalltau = sum_kendalltau / num_seeds
    avg_mae = sum_mae / num_seeds
    avg_rmse = sum_rmse / num_seeds
    avg_test_time = sum_test_time / num_seeds

    avg_confusion_matrix, avg_precisions, avg_recalls, avg_query_avgs, avg_query_stds, avg_obj_class_count_matrix, \
    avg_cls_precisions, avg_cls_recalls, precisions_matrix, recalls_matrix, precision_class_ranking, recall_class_ranking = metrics_calculator.get_metrics()

    test_results = {
        'seeds': seeds,
        'loss': avg_loss,
        'loss_dict': avg_loss_dict,
        'mae': avg_mae.detach().cpu().item(),
        'rmse': avg_rmse.detach().cpu().item(),
        'r2': avg_r2,
        'kendalltau': avg_kendalltau,
        'confusion_matrix_abs': avg_confusion_matrix.detach().cpu().numpy(),
        'confusion_matrix_rel': (avg_confusion_matrix/avg_confusion_matrix.sum(dim=-1, keepdim=True)).detach().cpu().numpy(),
        'accuracy-1': avg_acc.detach().cpu().item(),
        'precisions': avg_precisions.detach().cpu().numpy(),
        'avgPrecision': avg_precisions.mean().detach().cpu().item(),
        'avgPrecision2-4': avg_precisions[2:].mean().detach().cpu().item(),
        'recalls': avg_recalls.detach().cpu().numpy(),
        'avgRecall': avg_recalls.mean().detach().cpu().item(),
        'avgRecall2-4': avg_recalls[2:].mean().detach().cpu().item(),
        'time': avg_test_time
    }
    if avg_query_avgs is not None:
        test_results.update({
            'avgQueryAvgs': avg_query_avgs.detach().cpu().numpy(),
            'avgQueryStds': avg_query_stds.detach().cpu().numpy()
        })
    if avg_obj_class_count_matrix is not None:
        test_results.update({
            'avgObjClassCount': avg_obj_class_count_matrix.detach().cpu().numpy(),
            'avgClsPrecisions': avg_cls_precisions.detach().cpu().numpy(),
            'avgClsRecalls': avg_cls_recalls.detach().cpu().numpy(),
            'precisionsMatrix': precisions_matrix.detach().cpu().numpy(),
            'recallsMatrix': recalls_matrix.detach().cpu().numpy(),
            'precisionClassRanking': precision_class_ranking.detach().cpu().numpy(),
            'recallClassRanking': recall_class_ranking.detach().cpu().numpy()
        })
    if return_last_preds:
        return test_results, cur_preds
    else:
        return test_results


def evaluate_models_on_datasets(args, base_data_dir, eval_dataset: SupportedEvalDatasets, models, criteria, device,
                                num_workers=0, test_transforms=None, batch_size=128, num_count_classes=5,
                                seeds=None, disable_tqdm=False, logging=True, return_last_preds=False):

    if not isinstance(models, list):
        models = [models]
    if not isinstance(criteria, list):
        criteria = [criteria]

    if seeds is None:
        seeds = [None for _ in range(len(models))]

    if not isinstance(seeds, list):
        seeds = [seeds]

    if logging:
        LOGGER.info(f'Evaluating {models}')

    if eval_dataset is SupportedEvalDatasets.BaselineDefault:
        eval_datasets = [SupportedEvalDatasets.FSC147, SupportedEvalDatasets.FSC147_low,
                         SupportedEvalDatasets.FSC147_medium, SupportedEvalDatasets.FSC147_high]
    elif eval_dataset is SupportedEvalDatasets.EvalDefault:
        eval_datasets = [SupportedEvalDatasets.FSC147, SupportedEvalDatasets.FSC147_low]
    else:
        eval_datasets = [eval_dataset]

    results = [{} for _ in models]
    if return_last_preds:
        preds = [{} for _ in models]
    for eval_dataset in eval_datasets:
        test_dataset = create_eval_dataset(eval_dataset, base_data_dir, transform=test_transforms,
                                           max_num_obj=num_count_classes - 1, disable_tqdm=disable_tqdm,
                                           use_reference_crops=True, reference_crop_size=args.reference_crop_size,
                                           density_scaling=args.density_scaling, img_size=args.img_size)

        for i, (model, criterion) in enumerate(zip(models, criteria)):
            result = test_model(args, model, criterion, num_workers=num_workers, batch_size=batch_size, test_dataset=test_dataset, num_count_classes=num_count_classes,
                                device=device, seeds=seeds[i], return_last_preds=return_last_preds)
            if return_last_preds:
                result, cur_preds = result
                preds[i][eval_dataset.name] = cur_preds
            results[i][eval_dataset.name] = result
            if logging:
                if hasattr(criterion, 'num_obj_classes'):
                    if test_dataset.NUM_OBJ_CLASSES != criterion.num_obj_classes:
                        LOGGER.warning(f'The number of object classes the model was trained with ({criterion.num_obj_classes}) differs from {eval_dataset.name} ({test_dataset.NUM_OBJ_CLASSES}), the object class specific metrics are unreliable!')
                print_results = filter_result(result)
                LOGGER.info(f'{eval_dataset.name} test results {model}: {print_results}')
    if return_last_preds:
        return results, preds
    return results


def load_evaluation_results(model_dir):
    model_results_path = os.path.join(model_dir, RESULTS_FILE_NAME)
    results = None
    if os.path.isfile(model_results_path):
        LOGGER.info('Loading existing results')
        with open(model_results_path, 'rb') as f:
            results = torch.load(f)
    return results


def save_evaluation_results(model_results, model_dirs=None, summary_writers=None, model_dicts=None, epoch=None,
                            log_classification_metrics=True):

    if model_dirs is None and summary_writers is None:
        LOGGER.warning(f'model_dirs and summary_writers cannot be both None')
        return

    if summary_writers is None and epoch is not None:
        LOGGER.warning(f'{summary_writers=} while {epoch=}, ignoring save_evaluation_results call')
        return

    num_models = len(model_dirs) if model_dirs is not None else len(summary_writers)

    if model_dirs is None:
        model_dirs = [None for _ in range(num_models)]
    if summary_writers is None:
        summary_writers = [None for _ in range(num_models)]

    if model_dicts is None:
        model_dicts = [None for _ in range(num_models)]

    for model_dir, model_result, summary_writer, model_dict in zip(model_dirs, model_results, summary_writers, model_dicts):

        if epoch is None:
            model_results_path = os.path.join(model_dir, RESULTS_FILE_NAME)
            with open(model_results_path, 'wb') as f:
                torch.save(model_result, f)

            for key, value in model_result.items():
                with open(os.path.join(model_dir, f'{key}_confusion_matrix_abs.csv'), 'w') as f:
                    np.savetxt(f, value['confusion_matrix_abs'], delimiter=',')

            if summary_writer is not None and model_dict is not None:
                summary_result_dict = {}
                for dataset, results_dict in model_result.items():
                    summary_result_dict[f'hparam/{dataset}_loss'] = results_dict['loss']
                    for loss_name, loss_value in results_dict['loss_dict'].items():
                        summary_result_dict[f'hparam/{dataset}_{loss_name}'] = results_dict['loss_dict'][loss_name]
                    summary_result_dict[f'hparam/{dataset}_accuracy-1'] = results_dict['accuracy-1']
                    summary_result_dict[f'hparam/{dataset}_mae'] = results_dict['mae']
                    summary_result_dict[f'hparam/{dataset}_rmse'] = results_dict['rmse']
                    summary_result_dict[f'hparam/{dataset}_avgPrecision'] = results_dict['avgPrecision']
                    summary_result_dict[f'hparam/{dataset}_avgRecall'] = results_dict['avgRecall']
                    for i in range(results_dict['precisions'].shape[0]):
                        summary_result_dict[f'hparam/{dataset}_precision_{i}'] = results_dict['precisions'][i]
                        summary_result_dict[f'hparam/{dataset}_recall_{i}'] = results_dict['recalls'][i]
                    if 'avgQueryAvgs' in results_dict:
                        for i in range(results_dict['avgQueryAvgs'].shape[0]):
                            summary_result_dict[f'hparam/{dataset}_query_avg_{i}'] = results_dict['avgQueryAvgs'][i]
                            summary_result_dict[f'hparam/{dataset}_query_std_{i}'] = results_dict['avgQueryStds'][i]
                    if 'avgObjClassCount' in results_dict:
                        for i in range(results_dict['avgClsPrecisions'].shape[0]):
                            summary_result_dict[f'hparam/{dataset}_avgClsPrecision_{i}'] = results_dict['avgClsPrecisions'][i]
                            summary_result_dict[f'hparam/{dataset}_avgClsRecall_{i}'] = results_dict['avgClsRecalls'][i]

                summary_writer.add_hparams(prepare_dict_for_summary(model_dict), summary_result_dict)
        else:
            for dataset, results_dict in model_result.items():
                summary_writer.add_scalars('Loss', {f'{dataset}': results_dict['loss']}, epoch)
                for loss_name, loss_sum in results_dict['loss_dict'].items():
                    summary_writer.add_scalars(loss_name, {f'{dataset}': results_dict['loss_dict'][loss_name]}, epoch)
                if log_classification_metrics:
                    summary_writer.add_scalars('Accuracy', {f'{dataset}': results_dict['accuracy-1']}, epoch)
                    summary_writer.add_scalars('AvgPrecision', {f'{dataset}': results_dict['avgPrecision']}, epoch)
                    summary_writer.add_scalars('Precision', {f'{c}_{dataset}': results_dict['precisions'][c] for c in range(results_dict['precisions'].shape[0])}, epoch)
                    summary_writer.add_scalars('AvgRecall', {f'{dataset}': results_dict['avgRecall']}, epoch)
                    summary_writer.add_scalars('Recall', {f'{c}_{dataset}': results_dict['recalls'][c] for c in range(results_dict['recalls'].shape[0])}, epoch)
                summary_writer.add_scalars('MAE', {f'{dataset}': results_dict['mae']}, epoch)
                summary_writer.add_scalars('RMSE', {f'{dataset}': results_dict['rmse']}, epoch)


def filter_result(result):
    return {k: v for k, v in result.items() if
                     k not in ['precisionClassRanking', 'recallClassRanking', 'avgObjClassCount', 'precisionsMatrix',
                               'recallsMatrix']}
