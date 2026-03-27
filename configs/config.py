"""experiments configuration"""
import os.path as osp

from common.config.base import CN, _C

# public alias
cfg = _C
_C.VAL.METRIC = 'seg_iou'

# ---------------------------------------------------------------------------- #
# Datasets
# ---------------------------------------------------------------------------- #

_C.DATASET_TARGET = CN()
_C.DATASET_TARGET.TYPE = ''
_C.DATASET_TARGET.TEST = tuple()
_C.DATASET_TARGET.TRAIN = tuple()

# NuScenesLidarSegSCN
_C.DATASET_TARGET.NuScenesLidarSeg = CN()
_C.DATASET_TARGET.NuScenesLidarSeg.preprocess_dir = ''
_C.DATASET_TARGET.NuScenesLidarSeg.nuscenes_dir = ''
_C.DATASET_TARGET.NuScenesLidarSeg.merge_classes = True
_C.DATASET_TARGET.NuScenesLidarSeg.resize = (400, 225)

# SemanticKITTI
_C.DATASET_TARGET.SemanticKITTI = CN()
_C.DATASET_TARGET.SemanticKITTI.preprocess_dir = ''
_C.DATASET_TARGET.SemanticKITTI.semantic_kitti_dir = ''
_C.DATASET_TARGET.SemanticKITTI.merge_classes = True
_C.DATASET_TARGET.SemanticKITTI.crop_size =(480, 302)

_C.BMVC_INIT_VIEW = 0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# @ will be replaced by config path
_C.OUTPUT_DIR = osp.expanduser('output/@')