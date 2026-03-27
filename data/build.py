from torch.utils.data.dataloader import DataLoader
from yacs.config import CfgNode as CN

from data.collate import get_collate
from common.utils.torch_util import worker_init_fn
from data.nuscenes_lidarseg.nuscenes_lidarseg_dataloader import NuScenesLidarSeg
from data.semantic_kitti.semantic_kitti_dataloader import SemanticKITTI


def build_dataloader(cfg, mode='test', domain='target', different_batch_size=None):
    dataset_cfg = cfg.get('DATASET_' + domain.upper())
    split = dataset_cfg[mode.upper()]
    batch_size = cfg['VAL'].BATCH_SIZE
    if different_batch_size is not None:
        batch_size = different_batch_size

    # build dataset
    # Make a copy of dataset_kwargs so that we can pop augmentation afterward without destroying the cfg.
    dataset_kwargs = CN(dataset_cfg.get(dataset_cfg.TYPE, dict()))
    if dataset_cfg.TYPE == 'NuScenesLidarSeg':
        dataset = NuScenesLidarSeg(split=split, **dataset_kwargs)
    elif dataset_cfg.TYPE == 'SemanticKITTI':
        dataset = SemanticKITTI(split=split, **dataset_kwargs)
    else:
        raise ValueError('Unsupported type of dataset: {}.'.format(dataset_cfg.TYPE))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        collate_fn=get_collate
    )

    return dataloader
