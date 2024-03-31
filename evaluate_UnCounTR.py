"""
Based on https://github.com/Verg-Avesta/CounTR/blob/main/FSC_test_cross(few-shot).py
"""
import argparse
import datetime
import enum
import json
import logging
import os
import pathlib
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(os.path.join(cur_dir, 'src', 'third_party', 'CounTR', 'models_crossvit')))

from src.models.Baselines import FasterRCNNBaseline
from src.third_party.CounTR import models_mae_cross
from src.util.model_loader import load_model
from src.util.misc import get_filtered_model_dirs, match_str_and_enum
import src.util.transforms as custom_transforms
from src.data_handling.EvalDatasetFactory import create_eval_dataset
from src.data_handling.SupportedDatasets import SupportedEvalDatasets

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
LOGGER = logging.getLogger()


class EvalProtocol(enum.Enum):
    original = 0
    fixed_images = 1
    tiling_above_50 = 2


def parse_args():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    parser.add_argument('--debug', action='store_true', help='Log every prediction.')

    # test set
    parser.add_argument('--dataset_type', default='test', type=str, help='test, val, or MSO')
    parser.add_argument('--imgs', nargs='+', help='pass image names to evaluate')

    # inference
    parser.add_argument('--eval_protocol', default='original', type=str, help='Name of the evaluation protocol to use. See EvalProtocol for possible values.')
    parser.add_argument('--normalization', default=True, help='Set to False to disable test-time normalization')
    parser.add_argument('--norm_threshold', type=float, default=1.8, help='Threshold for test-time normalisation')
    parser.add_argument('--img_size', default=384, type=int, help='Image size used for evaluation')
    parser.add_argument('--stride', default=128, type=int, help='stride used for moving windows')
    parser.add_argument('--num_shots', default=3, type=int, help='Number of shots')
    parser.add_argument('--use_max_shots', action='store_true', help='use the maximum number of shots available')
    parser.add_argument('--max_count', default=0, type=int, help='only use images with less than max_count objects')
    parser.add_argument('--external', default=False, help='True if using external exemplars')
    parser.add_argument('--box_bound', default=-1, type=int, help='The max number of exemplars to be considered')

    # model
    parser.add_argument('--test_rcnn', action='store_true', help='Tests RCNN baseline.')
    parser.add_argument('--weights_dir', default='', type=str, help='path to directory with pretrained weights')
    parser.add_argument('--model_id', default=0, type=int, help='Id of the model to be loaded. If not specified, a trained CounTR model will be loaded.')
    parser.add_argument('--model_dir', default='', type=str, help='Path to a saved model. If not specified, a trained CounTR model will be loaded.')
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--density_scaling', default=60., type=float,
                        help='Scaling used for the density map. The total number of objects corresponding to a density map is derived by taking the sum of the map divided by this scaling.')

    # visualisation
    parser.add_argument('--num_visualisations', default=-1, type=int,
                        help='Number of samples that should be visualised. If -1, all samples will be visualised.')
    parser.add_argument('--show_visualisations', action='store_true', help='Show all visualisations.')
    parser.add_argument('--pred_visualisation_scaling', default=2., type=float, help='Scaling of the density map when visualising predictions.')

    # CounTR parameters
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # paths
    parser.add_argument('--data_path', default='./data/FSC147/', type=str, help='dataset path')
    parser.add_argument('--anno_file', default='annotation_FSC147_384.json', type=str, help='annotation json file')
    parser.add_argument('--data_split_file', default='Train_Test_Val_FSC_147.json', type=str,
                        help='data split json file')
    parser.add_argument('--im_dir', default='images_384_VarV2', type=str, help='images directory')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')

    # miscellaneous
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    args = parser.parse_args()

    args.eval_protocol = match_str_and_enum(args.eval_protocol, EvalProtocol)

    return args


def make_grid(imgs, h, w):
    assert len(imgs) == 9
    rows = []
    for i in range(0, 9, 3):
        row = torch.cat((imgs[i], imgs[i + 1], imgs[i + 2]), -1)
        rows += [row]
    grid = torch.cat((rows[0], rows[1], rows[2]), 0)
    grid = transforms.Resize((h, w))(grid.unsqueeze(0))
    return grid.squeeze(0)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    LOGGER.debug(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    LOGGER.debug(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        LOGGER.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class TestData(Dataset):
    def __init__(self, data_split, im_dir, annotations, external: bool, reference_crop_size, img_size, patch_size, box_bound: int = -1, normalise_transform=None,
                 max_count=0, density_scaling=1.0, split='test', imgs=None):

        self.img = data_split[split]
        self.img_dir = im_dir
        self.annotations = annotations
        self.external = external
        self.box_bound = box_bound
        self.normalise_transform = normalise_transform
        self.density_scaling = density_scaling
        self.reference_crop_size = reference_crop_size
        self.img_size = img_size
        self.patch_size = patch_size

        if imgs is not None and len(imgs) > 0:
            orig_len = len(self.img)
            filter_imgs = [f'{i}.jpg' for i in imgs]
            self.img = [i for i in self.img if i in filter_imgs]
            if len(imgs) != len(self.img):
                LOGGER.warning(f'Passed images {len(imgs)} but only {len(self.img)} found in dataset')
            LOGGER.info(f'Reduced length from {orig_len} to {len(self.img)}, passed images {len(imgs)}')

        if max_count > 0:
            filtered_imgs = [k for k, ann in self.annotations.items() if len(ann['points']) < max_count]
            self.img = [i for i in self.img if i in filtered_imgs]

        if external:
            self.external_boxes = []
            for anno in self.annotations:
                rects = []
                bboxes = self.annotations[anno]['box_examples_coordinates']

                if bboxes:
                    image = Image.open('{}/{}'.format(self.img_dir, anno))
                    image.load()
                    W, H = image.size

                    new_H = self.img_size
                    new_W = self.patch_size * int((W / H * self.img_size) / self.patch_size)
                    scale_factor_W = float(new_W) / W
                    scale_factor_H = float(new_H) / H
                    image = transforms.Resize((new_H, new_W))(image)
                    Normalize = transforms.Compose([transforms.ToTensor()])
                    image = Normalize(image)

                    for bbox in bboxes:
                        x1 = int(bbox[0][0] * scale_factor_W)
                        y1 = int(bbox[0][1] * scale_factor_H)
                        x2 = int(bbox[2][0] * scale_factor_W)
                        y2 = int(bbox[2][1] * scale_factor_H)
                        rects.append([y1, x1, y2, x2])

                    for box in rects:
                        box2 = [int(k) for k in box]
                        y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
                        bbox = image[:, y1:y2 + 1, x1:x2 + 1]
                        bbox = transforms.Resize((self.reference_crop_size, self.reference_crop_size))(bbox)
                        self.external_boxes.append(bbox.numpy())

            self.external_boxes = np.array(self.external_boxes if self.box_bound < 0 else
                                           self.external_boxes[:self.box_bound])
            self.external_boxes = torch.Tensor(self.external_boxes)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im_id = self.img[idx]
        anno = self.annotations[im_id]
        bboxes = anno['box_examples_coordinates'] if self.box_bound < 0 else \
            anno['box_examples_coordinates'][:self.box_bound]
        dots = np.array(anno['points'])

        image = Image.open('{}/{}'.format(self.img_dir, im_id))
        image.load()
        W, H = image.size

        new_H = self.img_size
        new_W = self.patch_size * int((W / H * self.img_size) / self.patch_size)
        scale_factor_W = float(new_W) / W
        scale_factor_H = float(new_H) / H
        image = transforms.Resize((new_H, new_W))(image)
        trans = [transforms.ToTensor()]
        if self.normalise_transform is not None:
            trans.append(self.normalise_transform)
        Normalize = transforms.Compose(trans)
        image = Normalize(image)

        boxes = []
        rel_box_coords = []
        if self.external:
            boxes = self.external_boxes
        else:
            rects = []
            for bbox in bboxes:
                x1 = int(bbox[0][0] * scale_factor_W)
                y1 = int(bbox[0][1] * scale_factor_H)
                x2 = int(bbox[2][0] * scale_factor_W)
                y2 = int(bbox[2][1] * scale_factor_H)
                rects.append([y1, x1, y2, x2])
                rel_box = torch.tensor([x1, y1, x2, y2], dtype=torch.float32) / torch.tensor([image.shape[-1], image.shape[-2], image.shape[-1], image.shape[-2]])
                rel_box_coords.append(rel_box)

            for box in rects:
                box2 = [int(k) for k in box]
                y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
                bbox = image[:, y1:y2 + 1, x1:x2 + 1]
                bbox = transforms.Resize((self.reference_crop_size, self.reference_crop_size))(bbox)
                boxes.append(bbox.numpy())

            boxes = np.array(boxes)
            boxes = torch.Tensor(boxes)

        rel_box_coords = torch.stack(rel_box_coords)

        if self.box_bound >= 0:
            assert len(boxes) <= self.box_bound

        # Only for visualisation purpose, no need for ground truth density map indeed.
        gt_map = np.zeros((image.shape[1], image.shape[2]), dtype='float32')
        for i in range(dots.shape[0]):
            gt_map[min(new_H - 1, int(dots[i][1] * scale_factor_H))][min(new_W - 1, int(dots[i][0] * scale_factor_W))] = 1
        gt_map = ndimage.gaussian_filter(gt_map, sigma=(1, 1), order=0)
        gt_map = torch.from_numpy(gt_map)
        gt_map = gt_map * self.density_scaling

        sample = {'image': image, 'dots': dots, 'boxes': boxes, 'pos': rects if self.external is False else [], 'gt_map': gt_map, 'name': im_id}
        return sample['image'], sample['dots'], sample['boxes'], sample['pos'], rel_box_coords, sample['gt_map'], sample['name']


class MSOWrapper:

    def __init__(self, args, data_path, transform, reference_crop_size, imgs=None):
        self.num_shots = args.num_shots
        self.reference_crop_size = reference_crop_size
        self.dataset = create_eval_dataset(SupportedEvalDatasets.MSO_few_shot, data_path.parent,
                                           transform=transform,
                                           disable_tqdm=True, use_reference_crops=True,
                                           reference_crop_size=self.reference_crop_size, density_scaling=args.density_scaling,
                                           img_size=args.img_size, num_shots=args.num_shots)
        self.filtered_idxs = None
        if imgs is not None and len(imgs) > 0:
            self.filtered_idxs = [int(i) for i in imgs]

    def __len__(self):
        if self.filtered_idxs is not None:
            return len(self.filtered_idxs)
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.filtered_idxs is not None:
            idx = self.filtered_idxs[idx]
        image, target_dict = self.dataset[idx]

        H, W = image.shape[1:]

        boxes = []
        rects = []
        unpad_boxes = self.dataset.unpad_sample(target_dict['boxes'])
        if unpad_boxes.shape[0] < self.num_shots:
            raise ValueError(f'Not enough boxes in image ({idx=})')
        for rel_bbox in unpad_boxes[:self.num_shots]:
            x1 = int(rel_bbox[0] * W)
            y1 = int(rel_bbox[1] * H)
            x2 = int(rel_bbox[2] * W)
            y2 = int(rel_bbox[3] * H)
            rects.append([y1, x1, y2, x2])

        for box in rects:
            box2 = [int(k) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            bbox = image[:, y1:y2 + 1, x1:x2 + 1]
            bbox = transforms.Resize((self.reference_crop_size, self.reference_crop_size))(bbox)
            boxes.append(bbox.numpy())

        boxes = np.array(boxes)
        boxes = torch.Tensor(boxes)

        rel_box_coords = unpad_boxes

        sample = {'image': image, 'dots': target_dict['global_count_targets'], 'boxes': boxes, 'pos': rects,
                  'gt_map': target_dict['global_count_targets'], 'name': f'mso_{idx}.png'}
        return sample['image'], sample['dots'], sample['boxes'], sample['pos'], rel_box_coords, sample['gt_map'], \
               sample['name']


def predict_using_tiling(args, model, image, boxes, num_shots, device, own_model):
    h, w = image.shape[-2:]
    third_w = int(w / 3)
    third_h = int(h / 3)
    remaining_w = w - third_w * 2
    remaining_h = h - third_h * 2
    r_images = []
    r_densities = []

    r_images.append(TF.crop(image[0], 0, 0, third_h, third_w))
    r_images.append(TF.crop(image[0], 0, third_w, third_h, third_w))
    r_images.append(TF.crop(image[0], 0, 2 * third_w, third_h, remaining_w))
    r_images.append(TF.crop(image[0], third_h, 0, third_h, third_w))
    r_images.append(TF.crop(image[0], third_h, third_w, third_h, third_w))
    r_images.append(TF.crop(image[0], third_h, 2 * third_w, third_h, remaining_w))
    r_images.append(TF.crop(image[0], 2 * third_h, 0, remaining_h, third_w))
    r_images.append(TF.crop(image[0], 2 * third_h, third_w, remaining_h, third_w))
    r_images.append(TF.crop(image[0], 2 * third_h, 2 * third_w, remaining_h, remaining_w))

    pred_cnt = 0
    for r_image in r_images:
        r_image = transforms.Resize((h, w))(r_image).unsqueeze(0)
        density_map, cur_pred_cnt = predict_with_sliding_window(args, model, r_image, boxes, num_shots, args.stride, device,
                                                                own_model)
        pred_cnt += cur_pred_cnt
        r_densities += [density_map]
    density_map = torch.concat([torch.concat(r_densities[tmp * 3:(tmp + 1) * 3], dim=1) for tmp in range(3)], dim=0)
    return density_map, pred_cnt


def predict_with_sliding_window(args, model, image, boxes, num_shots, stride, device, own_model):
    h, w = image.shape[-2:]
    density_map = torch.zeros([h, w], device=device)

    # keeps track of how many windows covered each pixel
    density_normalisation = torch.zeros([h, w], device=device, dtype=torch.int16)

    start = 0

    with torch.no_grad():
        while start + (args.img_size - 1) < w:
            try:
                if own_model:
                    output, = model(image[:, :, :, start:start + args.img_size], None, boxes, num_shots)
                else:
                    output, = model(image[:, :, :, start:start + args.img_size], boxes, num_shots)
            except AttributeError as e:
                LOGGER.error(
                    f'Error: Model does not support sliding window, probably because it uses relative box coordinates.')
                raise e
            output = output.squeeze(0)

            density_normalisation[:, start:start + args.img_size] += 1
            density_map[:, start:start + args.img_size] += output

            start = start + stride
            if start + (args.img_size - 1) >= w:
                if start == w - args.img_size + stride:
                    break
                else:
                    start = w - args.img_size

    density_map = density_map / density_normalisation
    pred_cnt = torch.sum(density_map / args.density_scaling).item()
    return density_map, pred_cnt


def predict_with_static_size(args, model, image, rel_box_coords, boxes, num_shots, img_size, own_model):
    if own_model:
        density_map = model(transforms.Resize((img_size, img_size))(image), rel_box_coords, boxes, num_shots)
    else:
        density_map = model(transforms.Resize((img_size, img_size))(image), boxes, num_shots)
    if args.test_rcnn:
        pred_cnt = density_map.item()
        density_map = torch.zeros(image.shape[-2:])
    else:
        pred_cnt = torch.sum(density_map / args.density_scaling).item()
        density_map = transforms.Resize(image.shape[-2:])(density_map).squeeze(0)

    return density_map, pred_cnt


def normalise_count(args, pred_cnt, density_map, exemplar_coords, coord_scaling=1):
    e_cnt = []
    for rect in exemplar_coords:
        e_cnt.append(torch.sum(density_map[coord_scaling * rect[0]:coord_scaling * rect[2] + 1,
                               coord_scaling * rect[1]:coord_scaling * rect[3] + 1] / args.density_scaling).item())
    if len(e_cnt) > 0:
        e_cnt = sum(e_cnt) / len(e_cnt)
        if e_cnt > args.norm_threshold:
            pred_cnt = pred_cnt/e_cnt
    return pred_cnt


def main(args):

    # load data
    data_path = Path(args.data_path)
    anno_file = data_path / args.anno_file
    data_split_file = data_path / args.data_split_file
    im_dir = data_path / args.im_dir

    with open(anno_file) as f:
        annotations = json.load(f)

    with open(data_split_file) as f:
        data_split = json.load(f)

    LOGGER.info('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    if args.debug:
        LOGGER.debug("{}".format(args).replace(', ', ',\n'))

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    subdir = args.eval_protocol.name
    subdir += f'_{args.img_size}'

    if args.norm_threshold != 1.8:
        subdir += f'_{args.norm_threshold}'

    if args.use_max_shots:
        subdir += '_max_shots'
    elif args.num_shots != 3:
        subdir += f'_{args.num_shots}'
    normalise_transform = None

    # paths
    base_dir = pathlib.Path(__file__).parent
    # models_dir
    models_dir = os.path.join(base_dir, 'runs')

    if args.output_dir == '':
        args.output_dir = os.path.join(base_dir, 'FSC_evaluation')

    args.dataset_type = args.dataset_type.lower()

    args.output_dir = os.path.join(args.output_dir, args.dataset_type, subdir)

    args.reference_crop_size = 64
    args.patch_size = 16

    if args.test_rcnn:
        model = FasterRCNNBaseline(args.img_size)
        args.output_dir = os.path.join(args.output_dir, str(model))
        own_model = True
    else:
        if args.model_id > 0:
            model_dirs, sub_dirs = get_filtered_model_dirs(models_dir, [str(args.model_id)], return_sub_dirs=True, include_baselines=False, check_beginning=True)
            if len(model_dirs) == 0:
                raise FileNotFoundError(f'Error: No model found with id {args.model_id}')
            if len(model_dirs) > 1:
                LOGGER.warning(f'Warning: Multiple models found with id {args.model_id} ({model_dirs}). Using the first one.')
            LOGGER.info(f'Overwriting model_dir with {model_dirs[0]}')
            args.model_dir = model_dirs[0]

        if args.model_dir == '':
            # monkey patch timm's Block init to load CounTR in new version
            from timm.models.vision_transformer import Block
            Block._old_init = Block.__init__
            def new_init(self, *args, qk_scale=None, **kwargs):
                self._old_init(*args, **kwargs)
            Block.__init__ = new_init
            model = models_mae_cross.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
            args.output_dir = os.path.join(args.output_dir, args.model)
            state_dict = torch.load(os.path.join(args.weights_dir, 'FSC147.pth'), map_location=device)['model']
            model.load_state_dict(state_dict)
            own_model = False
        else:
            model, model_args = load_model(args.model_dir, device, args.weights_dir)
            args.density_scaling = model_args.density_scaling
            args.output_dir = os.path.join(args.output_dir, Path(args.model_dir).name)
            args.reference_crop_size = model_args.reference_crop_size
            args.patch_size = model.patch_size
            if model is None:
                raise FileNotFoundError(f'Error: No model found in {args.model_dir}')
            normalise_transform = model.backbone.normalise_transform
            own_model = True

    only_subset = args.imgs is not None and len(args.imgs) > 0

    box_width = 2
    separate_pred = False

    if args.dataset_type == 'val':
        dataset_test = TestData(data_split, im_dir, annotations, args.external, args.reference_crop_size, args.img_size, args.patch_size,
                                args.box_bound, normalise_transform, max_count=args.max_count,
                                density_scaling=args.density_scaling, split='val', imgs=args.imgs)
    elif args.dataset_type == 'test':
        dataset_test = TestData(data_split, im_dir, annotations, args.external, args.reference_crop_size, args.img_size, args.patch_size,
                                args.box_bound, normalise_transform, max_count=args.max_count,
                                density_scaling=args.density_scaling, imgs=args.imgs)
    elif args.dataset_type == 'mso':
        transform = [custom_transforms.ToTensor(), custom_transforms.Resize((args.img_size, args.img_size))]
        if normalise_transform is not None:
            transform.append(normalise_transform)
        transform = custom_transforms.Compose(transform)
        if args.num_shots > 1:
            LOGGER.warning(f'Warning: num_shots > 1 not supported for MSO. Setting num_shots to 1.')
            args.num_shots = 1
        dataset_test = MSOWrapper(args, data_path, transform, args.reference_crop_size, imgs=args.imgs)

    # create output dir
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    sampler_test = torch.utils.data.RandomSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model.to(device)

    LOGGER.info(f"Start testing.")
    start_time = time.time()

    # test
    epoch = 0
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    # some parameters in training
    train_mae = 0
    train_rmse = 0

    loss_array = []
    gt_array = []
    pred_arr = []
    name_arr = []

    for data_iter_step, (samples, gt_dots, boxes, pos, rel_box_coords, gt_map, im_name) in \
            enumerate(metric_logger.log_every(data_loader_test, print_freq, header)):

        im_name = Path(im_name[0])
        samples = samples.to(device, non_blocking=True)
        gt_dots = gt_dots.to(device, non_blocking=True).half()
        boxes = boxes.to(device, non_blocking=True)
        rel_box_coords = rel_box_coords.to(device, non_blocking=True)
        if not args.use_max_shots:
            pos = pos[:args.num_shots]
            boxes = boxes[:, :args.num_shots]
            rel_box_coords = rel_box_coords[:, :args.num_shots]
        cur_num_shots = len(pos)
        _, _, h, w = samples.shape

        s_cnt = 0
        for rect in pos:
            if rect[2] - rect[0] < 10 and rect[3] - rect[1] < 10:
                s_cnt += 1

        use_sliding_window = not (args.eval_protocol is EvalProtocol.fixed_images or args.test_rcnn)
        tile_img = use_sliding_window and (s_cnt >= 1)
        if use_sliding_window:
            if tile_img:
                density_map, pred_cnt = predict_using_tiling(args, model, samples, boxes, cur_num_shots, device, own_model)
            else:
                density_map, pred_cnt = predict_with_sliding_window(args, model, samples, boxes, cur_num_shots, args.stride, device, own_model)
        else:
            density_map, pred_cnt = predict_with_static_size(args, model, samples, rel_box_coords, boxes, cur_num_shots, args.img_size, own_model)

        if args.normalization:
            if tile_img:
                coord_scaling = 3
            else:
                coord_scaling = 1
            pred_cnt = normalise_count(args, pred_cnt, density_map, pos, coord_scaling)

        if len(gt_dots.shape) == 1:
            gt_cnt = gt_dots.item()
        else:
            gt_cnt = gt_dots.shape[1]

        if args.eval_protocol is EvalProtocol.tiling_above_50:
            if pred_cnt > 50 and not tile_img:
                # if the image contains many objects, we use the tiling approach to refine the prediction
                density_map, pred_cnt = predict_using_tiling(args, model, samples, boxes, cur_num_shots, device,
                                                              own_model)
                pred_cnt = normalise_count(args, pred_cnt, density_map, pos, 3)
                tile_img = True
        cnt_err = abs(pred_cnt - gt_cnt)
        train_mae += cnt_err
        train_rmse += cnt_err ** 2

        if args.debug:
            LOGGER.debug(f'{data_iter_step}/{len(data_loader_test)}: pred_cnt: {pred_cnt},  gt_cnt: {gt_cnt},  error: {cnt_err},  AE: {cnt_err},  SE: {cnt_err ** 2}, id: {im_name.name}, s_cnt: {s_cnt >= 1}')

        loss_array.append(cnt_err)
        gt_array.append(gt_cnt)
        pred_arr.append(pred_cnt)
        name_arr.append(im_name.name)

        split_imgs = True

        # compute and save images
        visualise_sample = args.num_visualisations == -1 or (args.num_visualisations > 0 and data_iter_step < args.num_visualisations)
        if visualise_sample:
            if normalise_transform is not None:
                img = normalise_transform.reverse(samples[0])
            else:
                img = samples[0]
            box_map = torch.zeros([img.shape[1], img.shape[2]], device=device)
            if not args.external:
                for rect in pos:
                    for i in range(rect[2] - rect[0]):
                        a = min(rect[0] + i, img.shape[1] - 1)
                        b1 = min(rect[1], img.shape[2] - 1)
                        b2 = min(rect[3], img.shape[2] - 1)
                        box_map[a, b1 - box_width:b1 + box_width] = 1
                        box_map[a, b2 - box_width:b2 + box_width] = 1
                    for i in range(rect[3] - rect[1]):
                        a1 = min(rect[0], img.shape[1] - 1)
                        a2 = min(rect[2], img.shape[1] - 1)
                        b = min(rect[1] + i, img.shape[2] - 1)
                        box_map[a1 - box_width:a1 + box_width, b] = 1
                        box_map[a2 - box_width:a2 + box_width, b] = 1
                box_map = box_map.unsqueeze(0)
                box_map = torch.cat((torch.zeros_like(box_map), box_map, torch.zeros_like(box_map)))

            if tile_img:
                pred = transforms.Resize((h, w))(density_map.unsqueeze(0))
                pred = pred * (density_map.sum()/pred.sum())
            else:
                pred = density_map.unsqueeze(0)
            pred /= pred.max()
            pred = torch.cat((torch.zeros_like(pred), torch.zeros_like(pred), pred))
            orig_img = (img + box_map).clip(0, 1)
            combined_pred_img = img * 0.4 + pred * args.pred_visualisation_scaling
            if separate_pred:
                final_img = torch.cat((orig_img, combined_pred_img, pred), -1)
            else:
                final_img = torch.cat((orig_img, combined_pred_img), -1)
            final_img = final_img.permute(1, 2, 0).detach().cpu().clip(0, 1)
            combined_pred_img = combined_pred_img.permute(1, 2, 0).detach().cpu().clip(0, 1)
            h, w = final_img.shape[:2]
            factor = 50
            if split_imgs:
                h, w = orig_img.shape[1:]
                plt.figure(figsize=(w / factor, h / factor))
                plt.rcParams.update({'font.size': 30})
                plt.imshow(orig_img.permute(1, 2, 0).cpu())
                str_gt_cnt = f'GT = ' + str(round(gt_cnt))
                txt = plt.text(10, 40, str_gt_cnt, c='w',
                               fontsize='xx-large')
                txt.set_path_effects([PathEffects.withStroke(linewidth=6, foreground='k')])
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir,
                                         f'gt_{im_name.stem}__{round(gt_cnt)}{im_name.suffix}'), bbox_inches='tight')
                if args.show_visualisations:
                    plt.show()
                plt.close()
                plt.figure(figsize=(w / factor, h / factor))
                plt.imshow(combined_pred_img)
                str_pred_cnt = str(round(pred_cnt))
                txt = plt.text(10, 40, str_pred_cnt, c='w', fontsize='xx-large')
                txt.set_path_effects([PathEffects.withStroke(linewidth=6, foreground='k')])
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir,
                                         f'single_pred_{im_name.stem}__{round(gt_cnt)}{im_name.suffix}'), bbox_inches='tight')
                if args.show_visualisations:
                    plt.show()
                plt.close()
            else:
                plt.figure(figsize=(w / factor, h / factor))
                plt.rcParams.update({'font.size': 30})
                plt.imshow(final_img.cpu())
                str_pred_cnt = str(round(pred_cnt))
                str_gt_cnt = f'GT = ' + str(round(gt_cnt))
                txt = plt.text(w - 20 * len(str_pred_cnt), h - 20, str_pred_cnt, c='w', fontsize='xx-large')
                txt.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='k')])
                txt = plt.text(10, 40, str_gt_cnt, c='w',
                               fontsize='xx-large')
                txt.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='k')])
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir,
                                         f'pred_{im_name.stem}__{round(pred_cnt)}_{round(gt_cnt)}{im_name.suffix}'), bbox_inches='tight')
                if args.show_visualisations:
                    plt.show()
                plt.close()
            plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})
            plt.close()

        if device == 'cuda':
            torch.cuda.synchronize()

    log_stats = {'MAE': train_mae / (len(data_loader_test)),
                 'RMSE': (train_rmse / (len(data_loader_test))) ** 0.5}

    LOGGER.info('Current MAE: {:5.2f}, RMSE: {:5.2f} '.format(train_mae / (len(data_loader_test)), (
                train_rmse / (len(data_loader_test))) ** 0.5))

    if not only_subset:
        if args.output_dir:
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        plt.figure()
        plt.scatter(gt_array, loss_array)
        plt.xlabel('Ground Truth')
        plt.ylabel('Error')
        plt.savefig(os.path.join(args.output_dir, 'test_stat.png'))

        results = np.vstack([np.arange(data_iter_step+1)+1, np.array(name_arr), np.array(pred_arr)])
        np.savetxt(os.path.join(args.output_dir, f'results.csv'), results, '%s')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    LOGGER.info('Testing time {}'.format(total_time_str))


if __name__ == '__main__':
    args = parse_args()
    main(args)
