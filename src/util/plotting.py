import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

PLOT_FILE_EXT = ['svg']


def plot_boxes(boxes, fig, ax=None, **kwargs):
    plt_boxes = []
    for box in boxes:
        rect = Rectangle(box[:2], *(box[-2:] - box[:2]), fill=False)
        plt_boxes.append(rect)

    if 'facecolor' not in kwargs:
        kwargs['facecolor'] = 'None'
    if 'edgecolor' not in kwargs:
        kwargs['edgecolor'] = 'r'
    p = PatchCollection(plt_boxes, **kwargs)
    if ax is None:
        ax = fig.get_axes()[0]
    ax.add_collection(p)


def plot_confusion_matrix(confusion_matrix, ign_diagonal=True, file_path=None, return_figure=True, labels=None):

    fig = plt.figure(figsize=confusion_matrix.shape)
    plt.rcParams.update({'font.size': 20})
    cmap = plt.colormaps.get_cmap('Blues')
    tmp = confusion_matrix.copy()
    if ign_diagonal:
        np.fill_diagonal(tmp, 0)
    plt.imshow(tmp, cmap=cmap)
    if labels is None:
        labels = [i for i in range(confusion_matrix.shape[1])]
    plt.xticks(np.arange(tmp.shape[0]), labels=labels)
    plt.yticks(np.arange(tmp.shape[1]), labels=labels)

    # Loop over data dimensions and create text annotations.
    for i, row in enumerate(confusion_matrix):
        for j, cell in enumerate(row):
            txt = plt.text(j, i, cell, ha="center", va="center", color="k", fontsize='small')
            txt.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='w')])
    plt.ylabel('label')
    plt.xlabel('prediction')
    plt.tight_layout()
    plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})

    if file_path is not None:
        for file_ext in PLOT_FILE_EXT:
            plt.savefig(file_path+f'.{file_ext}')

    if return_figure:
        return fig
    else:
        plt.close()
