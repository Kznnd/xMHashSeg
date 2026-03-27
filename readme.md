# xMHashSeg: Cross-modal Hash Learning for Training-free Unsupervised LiDAR Semantic Segmentation

Code of xMHashSeg.

## Preparation

Tested in Linux with:

 - Pytorch 2.0.1
 - CUDA 11.8
 - Python 3.9.0
 - nuscenes-devkit

Install pointnet2_ops for Point-SANN
> cd Point_NN/pointnet2_ops_lib/
> pip install -e .
> cd -

Install packages for other requirements
> pip install -r requirements.txt

Readers also need to download the pre-trained model of [DAM](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) and place it in DAM/metric_depth/checkpoints folder.

## Dataset

Reader need to perform preprocessing to generate runnable data.

### nuScenes

 1. Please download the test dataset in Full dataset (v1.0) from the [NuScenes website](https://www.nuscenes.org/) and extract it.
 2. Please edit the script data/nuscenes_lidarseg/preprocess.py as follows and then run it.
	-   `root_dir` should point to the root directory of the NuScenes dataset.
	-   `out_dir` should point to the desired output directory to store the pickle files.

### SemanticKITTI

 1. Please download the files from the [SemanticKITTI website](http://semantic-kitti.org/dataset.html) and additionally the [color data](http://www.cvlibs.net/download.php?file=data_odometry_color.zip) from the [Kitti Odometry website](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). Extract everything into the same folder. Similar to NuScenes preprocessing, we save all points that project into the front camera image as well as the segmentation labels to a pickle file.
2. Please edit the script xmuda/data/semantic_kitti/preprocess.py as follows and then run it.
    -   `root_dir` should point to the root directory of the SemanticKITTI dataset.
    -   `out_dir` should point to the desired output directory to store the pickle files.

## Usage

Firstly, adjust the "preprocess_dir" and "dataset_dir" parameters in the configuration files (such as nuscene.yaml) under the ./configs folder to the preprocessed data path and root data path that you have placed.

### Testing

Run the test code with:

> \# test nuScenes dataset 
> 
> python -W ignore test.py --cfg=configs/nuscenes.yaml
> 
> \# test SemanticKITTI dataset
> 
> python -W ignore test.py --cfg=configs/semantickitti.yaml

