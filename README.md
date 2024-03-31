# [CVPR 2024] Learning to Count without Annotations

This repository contains the code for the paper "Learning to Count without Annotations" published at CVPR'24 and allows training counting models in a self-supervised manner using Self-Collages.
We provide an [example notebook](SelfCollages.ipynb) to experiment with the proposed method. Details can be found in the paper. [[`Paper`](https://arxiv.org/abs/2307.08727)]

<div>
  <img width="100%" alt="Learning to Count without Annotations illustration" src="images/arch.gif">
</div>

While recent supervised methods for reference-based object counting continue to improve the performance on benchmark datasets, they have to rely on small datasets due to the cost associated with manually annotating dozens of objects in images. We propose UnCounTR, a model that can learn this task without requiring any manual annotations. To this end, we construct "Self-Collages", images with various pasted objects as training samples, that provide a rich learning signal covering arbitrary object types and counts.
Our method builds on existing unsupervised representations and segmentation techniques to successfully demonstrate for the first time the ability of reference-based counting without manual supervision. 
Our experiments show that our method not only outperforms simple baselines and generic models such as FasterRCNN and DETR, but also matches the performance of supervised counting models in some domains.

# Qualitative results
<img alt="Qualitative results" src="images/qualitative_samples.svg">
We show predictions on four images from the FSC-147 test set, the green boxes represent the exemplars. Our predicted count is the sum of the density map rounded to the nearest integer.

The model is able to correctly predict the number of objects for clearly separated and slightly overlapping instances (see Subfigures **a** and **c**).
The model also successfully identifies the object type of interest, e.g. in Subfigure **b** the density map correctly highlights the strawberries rather than blueberries.
One limitation of our model is partial or occluded objects. For example, in Subfigure **d** the prediction missed a few burgers which are possibly the ones partially shown on the edge. However, partial or occluded objects are also challenging and ambiguous for humans.

# Structure
The construction process of the Self-Collages is implemented in [SelfCollageDataset.py](src/data_handling/datasets/SelfCollageDataset.py). The UnCounTR model is implemented in [UnCounTRModel.py](src/models/UnCounTRModel.py).
<details>
<summary>Details about the structure of this repository.</summary>

The general structure of this repository is as follows:
- [mask_generator.py](mask_generator.py)
  - generates the masks for object images
- [train_UnCounTR.py](train_UnCounTR.py)
  - trains the UnCounTR model
- [evaluate_UnCounTR.py](evaluate_UnCounTR.py)
  - evaluates the UnCounTR model
- [aggregate_evaluation_results.py](aggregate_evaluation_results.py)
  - aggregates evaluation results and computes metrics on FSC-147 subsets
- [test_baselines.py](test_baselines.py)
  - tests baselines on the FSC-147 dataset
- [self_supervised_semantic_counting.py](self_supervised_semantic_counting.py)
  - performs self-supervised semantic counting on given images using the UnCounTR model
- [SelfCollages.ipynb](SelfCollages.ipynb)
  - example notebook
- [visualise_predictions.py](visualise_predictions.py)
  - visualises the predictions of a model
- [src](src)
  - source code
- [env_files](env_files)
  - environment files for Anaconda environments
</details>

# Setup
To reproduce the results of the paper, the following steps are required:
1. Cloning this repository
2. Downloading datasets
3. Downloading pretrained weights
4. Cloning third-party code
5. Installing environments

## Downloading datasets
<details>
<summary>Details about the expected dataset structure.</summary>

The datasets are expected to be in the folder ```SelfCollages/data/```. Our code expects the following datasets:

### FSC-147 
The FSC-147 dataset [(link)](https://github.com/cvlab-stonybrook/LearningToCountEverything) should be placed in ```SelfCollages/data/FSC147_384_V2```. The folder should contain the following subfolders and files:
```
FSC147_384_V2
└── annotation_FSC147_384.json
└── ImageClasses_FSC147.txt
└── Train_Test_Val_FSC_147.json
└── gt_density_map_adaptive_384_VarV2
│   └── *.npy
│    ...
└── images_384_VarV2
    └── *.jpg
     ...
```

### MSO
The MSO dataset [(link)](https://www.kaggle.com/datasets/jessicali9530/mso-dataset) should be placed in ```SelfCollages/data/MSO```. The folder should contain the following subfolders and files:
```
MSO
└── imgIdx.mat
└── img
    └── *.jpg
     ...
```

### CARPK
The CARPK dataset [(link)](https://lafi.github.io/LPN/) should be placed in ```SelfCollages/data/CARPK```. The folder should contain the following subfolders and files:
```
CARPK
└── Annotations
│   └── *.txt
│    ...
└── Images
│   └── *.png
│    ...
└── ImageSets
    └── test.txt
    └── train.txt
```

### SUN397
The SUN397 dataset [(link)](https://vision.princeton.edu/projects/2010/SUN/) should be placed in ```SelfCollages/data/SUN397```. The folder should contain the following subfolders:
```
SUN397
└── Partitions
│   └── Testing_0*.txt
│    ...
└── SUN397
    └── a
        └── abbey
            └── *.jpg
             ...
         ...
     ...
```

### ImageNet
The ImageNet-1k dataset [(link)](https://www.image-net.org/) should be placed in ```SelfCollages/data/ImageNet```. The folder should contain the following subfolders:
```
ImageNet
└── ILSVRC2012_devkit_t12
│    ...
└── train
│   └── n*
│       └── *.JPEG
│        ...
│    ...
└── val
    └── n*
        └── *.JPEG
         ...
     ...
```

### Noise dataset
The noise dataset [(link)](https://github.com/mbaradad/learning_with_noise) is only needed to reproduce the ablation results. We are using the large-scale StyleGAN-Oriented dataset which should be placed in ```SelfCollages/data/noise_dataset/large_scale/stylegan-oriented```. The folder should contain multiple subfolders with the images.
```
stylegan-oriented
└── 00000
    └── *.jpg
         ...
     ...
```
</details>

## Downloading pretrained weights
To obtain object segmentations, we use [selfmask](https://github.com/NoelShin/selfmask). Download the pretrained weights for the model with 20 queries (file name: ```selfmask_nq20.pt```) and place the file in ```SelfCollages/data/```.

If you want to run the UnCounTR model with a pretrained Leopart backbone, download the model weights from [here](https://github.com/MkuuWaUjinga/leopart) and place them in ```SelfCollages/data/```. Specifically, the filenames should be ```leopart_vitb8.ckpt``` and ```leopart_vits16.ckpt```.

## Cloning third-party code
Create a directory called ```SelfCollages/src/third_party``` and clone the following repositories into it:
- [CounTR](https://github.com/Verg-Avesta/CounTR)
- [selfmask](https://github.com/NoelShin/selfmask) 

## Installing environments
To run this code, you need to install the Anaconda environment specified in [env.yml](env_files/env.yml) (for CPU only) or [env_gpu.yml](env_files/env_gpu.yml) (for GPU). This can be done using **one** of the following commands when in the root directory of this repository:

```conda env create -f env_files/env.yml``` or ```conda env create -f env_files/env_gpu.yml```

With the object mask generation being the only exception, all commands require the previously installed conda environment. It can be activated using: 
```
conda activate uncountr
```

To generate object masks, we use [selfmask](https://github.com/NoelShin/selfmask). Make sure to install the necessary requirements for this code as mentioned in the [corresponding repository](https://github.com/NoelShin/selfmask).


## Pretrained model
Instead of training UnCounTR from scratch, you can download the pretrained model [here](https://drive.google.com/file/d/1uc_kYNF0X8OSaiQWZJkFf5R082W_PxJc/view?usp=sharing).
Extract the zip-file and place the model folder in ```SelfCollages/runs/```. The folder should contain the following files:
```
uncountr_model
└── args.pt
└── UnCounTRModel.pt
```

# Training
All commands should be executed from the root directory of this repository. To get more information about optional arguments, use the ```--help``` flag.

To train UnCounTR from scratch, the object masks of the unlabelled ImageNet images have to be generated first followed by the training step itself.

**Generate object masks** This step creates object masks for the images in the ImageNet dataset which are saved in ```/path/to/SelfCollages/data/ImageNet/segmentations/selfmask```. Unlike all other commands, this step must be executed in the ```selfmask``` environment.
```
python mask_generator.py --data_dir=/path/to/SelfCollages/data --img_net_path=/path/to/SelfCollages/data/ImageNet
```

**Train UnCounTR model** This step trains the UnCounTR model using Self-Collages. The trained model will be saved in ```/path/to/SelfCollages/runs/model_name```.
```
python train_UnCounTR.py
```

# Evaluation
**Evaluate UnCounTR** The UnCounTR model can be tested on different datasets and subsets using:
```
python evaluate_UnCounTR.py  --model_dir=/path/to/SelfCollages/runs/model_name --data_path=/path/to/SelfCollages/data/FSC147_384_V2 --weights_dir=/path/to/SelfCollages/data --output_dir=/path/to/output_directory --dataset_type=dataset_type
```
<details>
<summary>details</summary>

```dataset_type``` can be ```test``` or ```val```, for the corresponding FSC-147 splits, or ```MSO```: This saves the evaluation results as well as visualisations for the predictions in the specified output directory.
</details>

**Aggregate results**
After evaluation, the following command allows to compute additional metrics and to calculate the performance on different subsets:  
```
python aggregate_evaluation_results.py --eval_results_path=/path/to/output_directory/dataset_type/subdir --dataset_type=dataset_type
```
<details>
<summary>details</summary>

```dataset_type``` can be ```test``` or ```val```, for the corresponding FSC-147 splits, or ```MSO```: This step uses the evaluation results saved in the output directory of the previous step to compute the results on FSC-147 subsets. The results are saved in a CSV file in  ```/path/to/SelfCollages/results```.
</details>

**Evaluate baselines** To compare UnCounTR to several baselines, they can be evaluated using: 
```
python test_baselines.py --img_size=384 --batch_size=32
```
<details>
<summary>details</summary>

This step evaluates the baselines on the FSC-147 dataset. The results for each baseline are saved in subfolders in ```/path/to/SelfCollages/runs/```. The results of all baselines for the different subsets is stored in a CSV file in ```/path/to/SelfCollages/results```.
</details>

# Self-supervised semantic counting
After training UnCounTR from scratch or downloading the pretrained model, it can be used for self-supervised semantic counting.
```
python self_supervised_semantic_counting.py --model_dir=/path/to/model --img_dir=/path/to/imgs
```
<details>
<summary>details</summary>

```img_dir``` should indicate the directory which contains the images of interest: This step performs self-supervised semantic counting on the images in the specified directory using the trained UnCounTR model. The results are saved in the same directory.
</details>

# Example notebook
The [example notebook](SelfCollages.ipynb) contains the commands described above to train and evaluate UnCounTR as well as the necessary steps to experiment with self-supervised semantic counting.
To start the notebook, use the following command:
```
jupyter notebook SelfCollages.ipynb
```

You might want to increase the maximum amount of memory that can be used by the notebook. This can be done with the argument ```--NotebookApp.max_buffer_size=X``` where ```X``` is the maximum amount of memory in bytes.

# Citation
If you find this repository useful, please consider citing our paper:
```
@article{knobel2024learning,
      title={Learning to Count without Annotations}, 
      author={Lukas Knobel and Tengda Han and Yuki M. Asano},
      journal={arXiv preprint arXiv:2307.08727},
      year={2024}
}
```

# Licensing
The code is licensed under the [MIT License](LICENSE.txt) except for code taken from other sources, where we specify the source at the beginning of the file. 
For the pretrained weights, please refer to the license specified in the [DINO repository](https://github.com/facebookresearch/dino). 
