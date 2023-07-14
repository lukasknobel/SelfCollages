import logging
import os
import pickle
import shutil
import tarfile
from pathlib import Path, PureWindowsPath, PurePath
from typing import Any, Dict, List, Optional, Tuple

import torch
from sklearn.cluster import KMeans
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import verify_str_arg, check_integrity
from tqdm import tqdm

from ...util import Constants
from ...util.SSLBoxAnnotator import SSLBoxAnnotator
from ...util.SegmentationsWrapper import SegmentationsWrapper
from ...util.misc_enums import ClusterFilter

META_FILE = "meta.bin"
LOGGER = logging.getLogger()


class CustomImageNet(ImageFolder):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    SUBSET_DESC_FILE_NAME = 'subsetfiles'

    NUM_OBJ_CLASSES = 1000

    def __init__(self, root: str, processed_meta_dir: Optional[str], split: str = "train", subset_size_factor=1,
                 return_segmentation_masks: bool = False, box_annotator: Optional[SSLBoxAnnotator] = None,
                 use_clusters: bool = False, num_clusters: int = 10000, feature_encoder=None,
                 filter_clusters: ClusterFilter = ClusterFilter.No, ign_split_subfolder=False, img_net_val_gt_path='', img_net_meta_file_path='',
                 **kwargs: Any) -> None:
        root = self.root = os.path.expanduser(root)
        self.ign_split_subfolder = ign_split_subfolder
        self.img_net_val_gt_path = img_net_val_gt_path
        self.img_net_meta_file_path = img_net_meta_file_path
        if self.ign_split_subfolder:
            LOGGER.warning(f'ign_split_subfolder set to True, make sure that the specified path {self.root} contains the images for {split}!')
        if processed_meta_dir is None:
            processed_meta_dir = root
        elif not os.path.isdir(processed_meta_dir):
            os.makedirs(processed_meta_dir)
        self.processed_meta_dir = os.path.expanduser(processed_meta_dir)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        self.parse_archives()
        wnid_to_classes = load_meta_file(self.processed_meta_dir)[0]

        # set to False and only set to True shortly before creating the segmentation wrapper
        self.return_segmentation_masks = False

        self.use_clusters = use_clusters
        self.num_clusters = num_clusters
        self.filter_clusters = filter_clusters

        if not self.use_clusters and self.filter_clusters is not ClusterFilter.No:
            raise ValueError('filter_clusters can only be used if use_clusters is True')

        super().__init__(self.split_folder, **kwargs)
        self.root = root
        self.targets = torch.tensor(self.targets)

        # create annotations before subset is created
        self.box_annotator = box_annotator
        self.boxes = None
        if self.box_annotator is not None:
            self.boxes = self.box_annotator.annotate_boxes(self, self.processed_meta_dir)

        self.feature_encoder = feature_encoder
        self.feature_representations = None
        self.clusters = None
        if self.use_clusters:
            if self.feature_encoder is None:
                raise ValueError('feature_encoder must be specified if use_clusters is True')
            self.feature_representations = self.feature_encoder.encode_dataset(self, self.processed_meta_dir)
            self.kmeans, self.clusters = self._get_clusters()
            self.targets = self.clusters
            # num_clusters x num_clusters similarity matrix
            self.cluster_similarity = self._get_cluster_similarity()
            # each row contains the indices of the most similar clusters (excluding the cluster itself)
            self.similar_cluster_ids = torch.argsort(self.cluster_similarity, -1, descending=True)[:, 1:]

        all_img_names_2_orig_idx = {Path(s[0]).name: i for i, s in enumerate(self.samples)}
        cur_idx_2_orig_idx = None

        use_subsets = subset_size_factor < 1
        if use_subsets:
            LOGGER.debug(f'Using subset of ImageNet')
            subset_desc_file_path = os.path.join(self.processed_meta_dir, CustomImageNet.SUBSET_DESC_FILE_NAME +f'_{subset_size_factor}_{self.split}.pt')
            if os.path.isfile(subset_desc_file_path):
                with open(subset_desc_file_path, 'rb') as f:
                    subset_samples_stored_paths, subset_targets = torch.load(f)

                # read in paths, make sure that both, stored windows and linux paths are supported
                if self.ign_split_subfolder:
                    subset_samples_current_paths = [os.path.join(self.root, *PurePath(PureWindowsPath(p[0]).as_posix()).parts[-2:]) for p in
                                                subset_samples_stored_paths]
                else:
                    subset_samples_current_paths = [
                        os.path.join(self.root, *PurePath(PureWindowsPath(p[0]).as_posix()).parts[-3:]) for p in subset_samples_stored_paths]
                subset_samples = sorted([(p, t) for p, t in zip(subset_samples_current_paths, subset_targets)])

            else:
                LOGGER.info(f'Getting new subset indices')
                subset_samples = []
                subset_targets = []
                for target in self.targets.unique():
                    target_idxs = (self.targets == target).nonzero().squeeze()
                    target_subset_size = int(target_idxs.shape[0] * subset_size_factor)
                    target_subset_idxs = target_idxs[torch.randperm(target_idxs.shape[0])[:target_subset_size]]
                    for target_subset_idx in target_subset_idxs:
                        subset_samples.append(self.samples[target_subset_idx])
                        subset_targets.append(self.targets[target_subset_idx])

                with open(subset_desc_file_path, 'wb') as f:
                    torch.save((subset_samples, subset_targets), f)
            missing_files = set([Path(s[0]).name for s in subset_samples]).difference(set([Path(s[0]).name for s in self.samples]))
            if len(missing_files) > 0:
                LOGGER.error(f'The following files are not in the directory: {missing_files}')
                raise FileNotFoundError(f'{len(missing_files)} files are missing in the directory')

            self.samples = subset_samples
            self.targets = torch.tensor(subset_targets)
            self.imgs = self.samples
            cur_idx_2_orig_idx = [all_img_names_2_orig_idx[Path(s[0]).name] for s in self.samples]

            if self.boxes is not None:
                self.boxes = self.boxes[cur_idx_2_orig_idx]

            if self.feature_representations is not None:
                self.feature_representations = self.feature_representations[cur_idx_2_orig_idx]

            if self.clusters is not None:
                self.clusters = self.clusters[cur_idx_2_orig_idx]
                self.targets = self.clusters

        self.unique_clusters, self.num_samples_per_cluster = self.targets.unique(return_counts=True)

        if self.filter_clusters is not ClusterFilter.No:
            prev_size = len(self.samples)

            if self.filter_clusters is ClusterFilter.Top10 or self.filter_clusters is ClusterFilter.Top20 or \
                    self.filter_clusters is ClusterFilter.Top50 or self.filter_clusters is ClusterFilter.Top100 or \
                    self.filter_clusters is ClusterFilter.Top200 or self.filter_clusters is ClusterFilter.Top500 or \
                    self.filter_clusters is ClusterFilter.Top2:
                # descending order
                sorted_cluster_idxs = torch.argsort(self.num_samples_per_cluster, descending=True)
                if self.filter_clusters is ClusterFilter.Top500:
                    n = 500
                elif self.filter_clusters is ClusterFilter.Top200:
                    n = 200
                elif self.filter_clusters is ClusterFilter.Top100:
                    n = 100
                elif self.filter_clusters is ClusterFilter.Top50:
                    n = 50
                elif self.filter_clusters is ClusterFilter.Top20:
                    n = 20
                elif self.filter_clusters is ClusterFilter.Top2:
                    n = 2
                else:
                    n = 10
                self.unique_clusters = self.unique_clusters[sorted_cluster_idxs[:n]]
                self.num_samples_per_cluster = self.num_samples_per_cluster[self.unique_clusters.to(torch.int64)]

            elif self.filter_clusters is ClusterFilter.AtLeast150:
                self.unique_clusters, filter_mask = self.get_filtered_clusters(150, return_mask=True)
                self.num_samples_per_cluster = self.num_samples_per_cluster[filter_mask]
            else:
                raise NotImplementedError(f'Cluster filter {self.filter_clusters.name} not implemented')

            # get indices of samples that belong to selected clusters
            cluster_filter = self.clusters.clone()
            cluster_filter.apply_(lambda x: x in self.unique_clusters)
            cur_idx_2_orig_idx = torch.arange(len(self.samples))[cluster_filter.to(torch.bool)]

            self._reduce_attributes(cur_idx_2_orig_idx)

            LOGGER.debug(f'Filtering clusters reduced the number of samples from {prev_size} to {len(self.samples)}')

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        if self.use_clusters and self.clusters is not None:
            self.classes = [[f'cluster_{i}'] for i in range(self.num_clusters)]
        else:
            self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

        self.return_segmentation_masks = return_segmentation_masks
        if self.return_segmentation_masks:
            if isinstance(cur_idx_2_orig_idx, torch.Tensor):
                cur_idx_2_orig_idx = [i.item() for i in cur_idx_2_orig_idx]
            self.seg_wrapper = SegmentationsWrapper(self.root, self.processed_meta_dir, len(self),
                                                    subset_idxs=cur_idx_2_orig_idx, in_memory=True, assume_sorted_subidxs=not use_subsets)

    def _reduce_attributes(self, selected_idxs):
        self.samples = [self.samples[sel_idx] for sel_idx in selected_idxs]
        self.targets = self.targets[selected_idxs]
        self.imgs = self.samples

        if self.boxes is not None:
            self.boxes = self.boxes[selected_idxs]

        if self.feature_representations is not None:
            self.feature_representations = self.feature_representations[selected_idxs]

        if self.clusters is not None:
            self.clusters = self.clusters[selected_idxs]
            self.targets = self.clusters

    def get_filtered_clusters(self, min_num_objs, return_mask=False):
        cluster_mask = self.num_samples_per_cluster >= min_num_objs
        if return_mask:
            return self.unique_clusters[cluster_mask], cluster_mask
        return self.unique_clusters[cluster_mask]

    def _get_clusters(self):
        clusters_dir = os.path.join(self.processed_meta_dir, 'clusters')
        os.makedirs(clusters_dir, exist_ok=True)

        kmeans_path = os.path.join(clusters_dir, f'kmeans_{self.num_clusters}.pt')
        clusters_path = os.path.join(clusters_dir, f'clusters_{self.num_clusters}.pt')

        clusters_stored = os.path.isfile(clusters_path)
        kmeans_stored = os.path.isfile(kmeans_path)
        loaded_clusters = False
        if clusters_stored and not kmeans_stored:
            raise FileNotFoundError(f'Clusters file {clusters_path} found but not kmeans file {kmeans_path}')
        if kmeans_stored:
            # load stored kmeans
            LOGGER.debug(f'Loading kmeans from {kmeans_path}')
            with open(kmeans_path, 'rb') as f:
                kmeans = pickle.load(f)
            if clusters_stored:
                LOGGER.debug(f'Loading clusters from {clusters_path}')
                with open(clusters_path, 'rb') as f:
                    clusters = torch.load(f)
                loaded_clusters = True
            else:
                # predict clusters with stored kmeans
                LOGGER.debug(f'Predicting clusters with stored kmeans')
                with open(kmeans_path, 'rb') as f:
                    kmeans = pickle.load(f)
                clusters = kmeans.predict(self.feature_representations)
        else:
            # train kmeans and predict clusters
            LOGGER.debug(f'Fitting new kmeans with {self.num_clusters} clusters')
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_init='auto',
                            verbose=True)  # , algorithm='elkan')
            clusters = kmeans.fit_predict(self.feature_representations)
            with open(kmeans_path, 'wb') as f:
                pickle.dump(kmeans, f)
        if not loaded_clusters:
            clusters = torch.from_numpy(clusters)
            with open(clusters_path, 'wb') as f:
                torch.save(clusters, f)
        return kmeans, clusters

    def _get_cluster_similarity(self, cosine=False, dim=-1, eps=1e-8):
        LOGGER.debug(f'Computing cluster similarity')

        cluster_centers = torch.from_numpy(self.kmeans.cluster_centers_)
        if cosine:
            numerator = cluster_centers @ cluster_centers.T
            cluster_norms = torch.norm(cluster_centers, dim=dim)
            denominator = torch.max(torch.mul(cluster_norms.unsqueeze(0), cluster_norms.unsqueeze(1)), torch.tensor(eps))
            cluster_similarity = torch.div(numerator, denominator)
        else:
            cluster_similarity = -torch.cdist(cluster_centers, cluster_centers, p=2)
        return cluster_similarity

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.use_clusters and self.clusters is not None:
            target = {Constants.TARGET_DICT_CLASSES: self.clusters[index]}
        else:
            target = {Constants.TARGET_DICT_CLASSES: target}

        if self.feature_representations is not None:
            target[Constants.TARGET_DICT_FEATURE_REPR] = self.feature_representations[index]

        if self.boxes is not None:
            target[Constants.TARGET_DICT_BOXES] = self.boxes[index]

        if self.return_segmentation_masks:
            target[Constants.TARGET_DICT_SEGMENTATIONS] = self.seg_wrapper.get_segmentation(index, sample.width, sample.height)
        if self.transform is not None:
            sample, target = self.transform(sample, target)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def parse_archives(self) -> None:
        self.parse_devkit_dir()
        if self.split == 'val':
            self.parse_val_archive()

    @property
    def split_folder(self) -> str:
        if self.ign_split_subfolder:
            return self.root
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def create_subset_tars(self, target_dir):
        # creates tar files for the selected subset
        train_subdir = os.path.join(target_dir, self.split)
        if not os.path.isdir(train_subdir):
            os.makedirs(train_subdir)

        for target in tqdm(self.targets.unique()):
            target_idxs = (self.targets == target).nonzero().squeeze()
            with tarfile.open(os.path.join(train_subdir, f'{self.wnids[target]}.tar'), 'w', dereference=True) as tarhandle:
                for target_idx in target_idxs:
                    sample_path = self.samples[target_idx][0]
                    tarhandle.add(sample_path, arcname=os.path.join(*Path(sample_path).parts[-2:]))

    def parse_devkit_dir(self) -> None:
        """Parse the devkit directory of the ImageNet2012 classification dataset and save
        the meta information in a binary file.
        """
        import scipy.io as sio

        def parse_meta_mat(devkit_root: str) -> Tuple[Dict[int, str], Dict[str, Tuple[str, ...]]]:
            if self.img_net_meta_file_path != '':
                metafile = self.img_net_meta_file_path
            else:
                metafile = os.path.join(devkit_root, "data", "meta.mat")
            meta = sio.loadmat(metafile, squeeze_me=True)["synsets"]
            nums_children = list(zip(*meta))[4]
            meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]
            idcs, wnids, classes = list(zip(*meta))[:3]
            classes = [tuple(clss.split(", ")) for clss in classes]
            idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
            wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
            return idx_to_wnid, wnid_to_classes

        def parse_val_groundtruth_txt(devkit_root: str) -> List[int]:
            if self.img_net_val_gt_path == '':
                file = os.path.join(devkit_root, "data", "ILSVRC2012_validation_ground_truth.txt")
            else:
                file = self.img_net_val_gt_path
            with open(file) as txtfh:
                val_idcs = txtfh.readlines()
            return [int(val_idx) for val_idx in val_idcs]

        devkit_root = os.path.join(self.root, "ILSVRC2012_devkit_t12")

        idx_to_wnid, wnid_to_classes = parse_meta_mat(devkit_root)
        val_idcs = parse_val_groundtruth_txt(devkit_root)
        val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

        torch.save((wnid_to_classes, val_wnids), os.path.join(self.processed_meta_dir, META_FILE))

    def parse_val_archive(
            self, wnids: Optional[List[str]] = None, folder: str = "val"
    ) -> None:
        """Parse the validation images directory of the ImageNet2012 classification dataset
        and prepare it for usage with the ImageNet dataset.

        Args:
            wnids (list, optional): List of WordNet IDs of the validation images. If None
                is given, the IDs are loaded from the meta file in the root directory
            folder (str, optional): Optional name for validation images folder. Defaults to
                'val'
        """
        val_root = os.path.join(self.root, folder)

        images = sorted(os.path.join(val_root, image) for image in os.listdir(val_root))

        if any([os.path.isdir(image_path) for image_path in images]):
            LOGGER.info('Found subdirectory in validation folder, assuming parsed validation data')
            return

        if wnids is None:
            wnids = load_meta_file(self.processed_meta_dir)[1]

        for wnid in set(wnids):
            os.mkdir(os.path.join(val_root, wnid))

        for wnid, img_file in zip(wnids, images):
            shutil.move(img_file, os.path.join(val_root, wnid, os.path.basename(img_file)))


def load_meta_file(root: str, file: Optional[str] = None) -> Tuple[Dict[str, str], List[str]]:
    if file is None:
        file = META_FILE
    file = os.path.join(root, file)

    if check_integrity(file):
        return torch.load(file)
    else:
        msg = (
            "The meta file {} is not present in the root directory or is corrupted. "
            "This file is automatically created by the ImageNet dataset."
        )
        raise RuntimeError(msg.format(file, root))
