import logging

import torch

LOGGER = logging.getLogger()


class MetricCalculator:
    """Keeps track of metrics"""
    def __init__(self):
        self.class_metrics = ClassMetricsCalculator()
        self.query_metrics = QueryMetricsCalculator()
        self.obj_class_counts = ObjClassCountMetricsCalculator()

    def update(self, raw_preds, loss_dict, confusion_matrix):
        self.class_metrics.update(confusion_matrix)
        self.query_metrics.update(raw_preds)
        self.obj_class_counts.update(loss_dict)

    def merge(self, metric_calculator):
        self.class_metrics.merge(metric_calculator.class_metrics)
        self.query_metrics.merge(metric_calculator.query_metrics)
        self.obj_class_counts.merge(metric_calculator.obj_class_counts)

    def get_metrics(self):
        cls_metrics = self.class_metrics.get_metrics()
        query_metrics = self.query_metrics.get_metrics()
        obj_metrics = self.obj_class_counts.get_metrics()
        return cls_metrics + query_metrics + obj_metrics

    def log_in_summary_writer(self, summary_writer, epoch):
        if self.query_metrics.has_data():
            query_preds_avg, query_preds_std = self.query_metrics.get_metrics()
            summary_writer.add_scalars('QueryPred_std', {f'query_{query_idx}': query_preds_std[query_idx] for query_idx in range(query_preds_std.shape[0])}, epoch)
            summary_writer.add_scalars('QueryPred_avg', {f'query_{query_idx}': query_preds_avg[query_idx] for query_idx in range(query_preds_avg.shape[0])}, epoch)

        if self.obj_class_counts.has_data():
            obj_class_count_matrix, avg_precision, avg_recall, _, _, _, _ = self.obj_class_counts.get_metrics()
            summary_writer.add_scalars('ClassAvgPrecision', {f'count_class_{cls_idx}': avg_precision[cls_idx] for cls_idx in range(avg_precision.shape[0])}, epoch)
            summary_writer.add_scalars('ClassAvgRecall', {f'count_class_{cls_idx}': avg_recall[cls_idx] for cls_idx in range(avg_recall.shape[0])}, epoch)


class ClassMetricsCalculator:
    """Keeps track of per-class metrics"""

    def __init__(self):
        self.confusion_matrix = None
        self.merged_calculators = []

    def merge(self, class_metrics_calculator):
        self.merged_calculators.append(class_metrics_calculator)

    def update(self, confusion_matrix):
        if self.confusion_matrix is None:
            self.confusion_matrix = torch.zeros_like(confusion_matrix)
        self.confusion_matrix += confusion_matrix

    def get_metrics(self):
        precisions = self.confusion_matrix.diagonal() / self.confusion_matrix.sum(dim=0)
        precisions[precisions.isnan()] = 0
        recalls = self.confusion_matrix.diagonal() / self.confusion_matrix.sum(dim=1)
        recalls[recalls.isnan()] = 0

        final_output = [
            self.confusion_matrix,
            precisions,
            recalls
        ]
        num_merges = len(self.merged_calculators)
        if num_merges > 0:
            for merged_calculator in self.merged_calculators:
                merged_output = merged_calculator.get_metrics()
                final_output[0] += merged_output[0]
                final_output[1] += merged_output[1]
                final_output[2] += merged_output[2]

            final_output[0] /= (num_merges+1)
            final_output[1] /= (num_merges+1)
            final_output[2] /= (num_merges+1)

        return tuple(final_output)


class QueryMetricsCalculator:
    """Keeps track of per-query metrics"""
    def __init__(self):
        self.query_preds = []
        self.merged_calculators = []

    def merge(self, query_metrics_calculator):
        if query_metrics_calculator.has_data():
            self.merged_calculators.append(query_metrics_calculator)

    def has_data(self):
        return len(self.query_preds) > 0

    def update(self, raw_preds):
        if isinstance(raw_preds, dict) and len(raw_preds['pred_logits'].shape) == 3:
            self.query_preds.append(raw_preds['pred_logits'].detach().cpu().argmax(-1).to(torch.float16))

    def get_metrics(self):
        if not self.has_data():
            return None, None

        query_preds = torch.concat(self.query_preds).to(torch.float64)
        query_preds_std = query_preds.std(0)
        query_preds_avg = query_preds.mean(0)

        final_output = [
            query_preds_avg,
            query_preds_std
        ]
        num_merges = len(self.merged_calculators)
        if num_merges > 0:
            for merged_calculator in self.merged_calculators:
                merged_output = merged_calculator.get_metrics()
                final_output[0] += merged_output[0]
                final_output[1] += merged_output[1]

            final_output[0] /= (num_merges+1)
            final_output[1] /= (num_merges+1)

        return tuple(final_output)


class ObjClassCountMetricsCalculator:
    """Keeps track of per object class and count metrics"""

    def __init__(self):
        self.obj_class_count_matrix = None
        self.merged_calculators = []

    def has_data(self):
        return self.obj_class_count_matrix is not None

    def update(self, loss_dict):
        if isinstance(loss_dict, dict) and 'obj_class_count_matrix' in loss_dict:
            obj_class_count_matrix = loss_dict['obj_class_count_matrix']
            if self.obj_class_count_matrix is None:
                self.obj_class_count_matrix = torch.zeros_like(obj_class_count_matrix)
            self.obj_class_count_matrix += obj_class_count_matrix

    def merge(self, obj_class_count_metrics_calculator):
        if obj_class_count_metrics_calculator.has_data():
            self.merged_calculators.append(obj_class_count_metrics_calculator)

    def get_metrics(self):
        if not self.has_data():
            return None, None, None, None, None, None, None
        precisions_matrix = self.obj_class_count_matrix[..., 0] / (
                self.obj_class_count_matrix[..., 0] + self.obj_class_count_matrix[..., 2])
        recall_matrix = self.obj_class_count_matrix[..., 0] / (
                self.obj_class_count_matrix[..., 0] + self.obj_class_count_matrix[..., 1])

        # filter out object classes without any data
        occurring_obj_class_count_matrix = self.obj_class_count_matrix[
            ~(self.obj_class_count_matrix == 0).flatten(1).all(1)]
        filtered_precision_matrix = occurring_obj_class_count_matrix[..., 0] / (
                occurring_obj_class_count_matrix[..., 0] + occurring_obj_class_count_matrix[..., 2])
        filtered_recall_matrix = occurring_obj_class_count_matrix[..., 0] / (
                occurring_obj_class_count_matrix[..., 0] + occurring_obj_class_count_matrix[..., 1])
        num_obj_classes = occurring_obj_class_count_matrix.shape[0]
        avg_precision = filtered_precision_matrix.nanmean(0)
        avg_recall = filtered_recall_matrix.nanmean(0)
        # precisions[precisions.isnan()] = 0
        # recalls[recalls.isnan()] = 0

        # get ranking of classes, sort ascending based on the negated matrices to ensure that nan values are treated as the lowest rather than the highest values
        precision_class_ranking = torch.argsort(-precisions_matrix, dim=0)
        recall_class_ranking = torch.argsort(-recall_matrix, dim=0)

        final_output = [
            self.obj_class_count_matrix,
            avg_precision,
            avg_recall,
            precisions_matrix,
            recall_matrix,
            precision_class_ranking,
            recall_class_ranking
        ]
        num_merges = len(self.merged_calculators)
        if num_merges > 0:
            for merged_calculator in self.merged_calculators:
                merged_output = merged_calculator.get_metrics()
                final_output[0] += merged_output[0]
                final_output[1] += merged_output[1]
                final_output[2] += merged_output[2]

            LOGGER.warning('Rankings wont be merged, using the first ranking')
            final_output[0] /= (num_merges+1)
            final_output[1] /= (num_merges+1)
            final_output[2] /= (num_merges+1)

        return tuple(final_output)
