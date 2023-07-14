import logging

from matplotlib import pyplot as plt
from torch.utils.data._utils import collate

LOGGER = logging.getLogger()


class PredictionVisualiser:
    def __init__(self, samples, model, name, **kwargs):
        self.imgs, self.targets = collate.default_collate(samples)
        self.name = name
        self.model = model
        self.kwargs = kwargs
        if not hasattr(self.model, 'visualise_prediction'):
            LOGGER.warning(f'Model does not have a visualise_prediction method. Cannot visualise predictions.')
            self.model = None

    def visualise_prediction(self, summary_writer=None, epoch=-1):
        if self.model is not None:
            fig = self.model.visualise_prediction(self.imgs, self.targets, return_fig=True, **self.kwargs)
            if summary_writer is not None:
                summary_writer.add_figure(f'{self.name}_predictions', fig, global_step=epoch)
            else:
                plt.show()
