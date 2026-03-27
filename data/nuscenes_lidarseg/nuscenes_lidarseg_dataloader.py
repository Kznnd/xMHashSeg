import os.path as osp
import pickle

from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class NuScenesLidarSegBase(Dataset):
    """NuScenes dataset"""

    class_names = [
        'ignore',
        'barrier',
        'bicycle',
        'bus',
        'car',
        'construction_vehicle',
        'motorcycle',
        'pedestrian',
        'traffic_cone',
        'trailer',
        'truck',
        'driveable_surface',
        'other_flat',
        'sidewalk',
        'terrain',
        'manmade',
        'vegetation'
    ]

    # UDA
    categories = {
        "vehicle": ["bicycle", "bus", "car", "construction_vehicle", "motorcycle", "trailer", "truck"],
        "driveable surface": ["driveable_surface"],
        "sidewalk": ["sidewalk"],
        "terrain": ["terrain"],
        "manmade": ["manmade"],
        "vegetation": ["vegetation"],
    }

    # zero-shot
    # categories = {
    #     "motorcycle": ["motorcycle"],
    #     "trailer": ["trailer"],
    #     "terrain": ["terrain"],
    #     "traffic_cone": ["traffic_cone"],
    #     "bicycle": ["bicycle"],
    #     "car": ["car"],
    # }

    def __init__(self,
                 split,
                 preprocess_dir,
                 merge_classes=False
                 ):

        self.split = split
        self.preprocess_dir = preprocess_dir

        print("Initialize Nuscenes dataloader")

        assert isinstance(split, tuple)
        print('Load', split)
        self.data = []
        for curr_split in split:
            with open(osp.join(self.preprocess_dir, curr_split + '.pkl'), 'rb') as f:
                self.data.extend(pickle.load(f))

        if merge_classes:
            self.label_mapping = -100 * np.ones(len(self.class_names), dtype=int)
            for cat_idx, cat_list in enumerate(self.categories.values()):
                for class_name in cat_list:
                    self.label_mapping[self.class_names.index(class_name)] = cat_idx
            self.class_names = list(self.categories.keys())
        else:
            self.label_mapping = None

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class NuScenesLidarSeg(NuScenesLidarSegBase):
    def __init__(self,
                 split,
                 preprocess_dir,
                 nuscenes_dir='',
                 merge_classes=False,
                 resize=(400, 225)
                 ):
        super().__init__(split,
                         preprocess_dir,
                         merge_classes=merge_classes
                         )

        self.nuscenes_dir = nuscenes_dir
        self.resize = resize

    def __getitem__(self, index):
        data_dict = self.data[index]
        num_classes = len(self.categories)

        pts_cam_coord = data_dict['pts_cam_coord']  # (N,3)

        seg_label = data_dict['seg_labels'].astype(np.int64)
        if self.label_mapping is not None:
            seg_label = self.label_mapping[seg_label]
        out_dict = {}
        points_img = data_dict['points_img']
        img_path = osp.join(self.nuscenes_dir, data_dict['camera_path'])
        image = Image.open(img_path)

        if self.resize:
            if not image.size == self.resize:
                # check if we do not enlarge downsized images
                assert image.size[0] > self.resize[0]

                # scale image points
                points_img[:, 0] = float(self.resize[1]) / image.size[1] * np.floor(points_img[:, 0])
                points_img[:, 1] = float(self.resize[0]) / image.size[0] * np.floor(points_img[:, 1])

                # resize image
                image = image.resize(self.resize, Image.BILINEAR)

        img_indices = points_img.astype(np.int64)

        assert np.all(img_indices[:, 0] >= 0)
        assert np.all(img_indices[:, 1] >= 0)
        assert np.all(img_indices[:, 0] < image.size[1])
        assert np.all(img_indices[:, 1] < image.size[0])

        # PIL to numpy
        image = np.asarray(image, dtype=np.float32) / 255.

        out_dict['seg_label'] = seg_label[seg_label >= 0]
        out_dict['img_indices'] = img_indices[seg_label >= 0]
        out_dict['pc'] = pts_cam_coord[seg_label >= 0]
        out_dict['img'] = image
        out_dict['num_classes'] = num_classes

        return out_dict
