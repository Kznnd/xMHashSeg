import os.path as osp
import pickle
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class SemanticKITTIBase(Dataset):
    """SemanticKITTI dataset"""

    # https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
    id_to_class_name = {
        0: "unlabeled",
        1: "outlier",
        10: "car",
        11: "bicycle",
        13: "bus",
        15: "motorcycle",
        16: "on-rails",
        18: "truck",
        20: "other-vehicle",
        30: "person",
        31: "bicyclist",
        32: "motorcyclist",
        40: "road",
        44: "parking",
        48: "sidewalk",
        49: "other-ground",
        50: "building",
        51: "fence",
        52: "other-structure",
        60: "lane-marking",
        70: "vegetation",
        71: "trunk",
        72: "terrain",
        80: "pole",
        81: "traffic-sign",
        99: "other-object",
        252: "moving-car",
        253: "moving-bicyclist",
        254: "moving-person",
        255: "moving-motorcyclist",
        256: "moving-on-rails",
        257: "moving-bus",
        258: "moving-truck",
        259: "moving-other-vehicle",
    }

    class_name_to_id = {v: k for k, v in id_to_class_name.items()}

    # merging classes
    # 10 class
    # categories = {
    #     'car': ['car', 'moving-car'],
    #     'truck': ['truck', 'moving-truck'],
    #     'bike': ['bicycle', 'motorcycle', 'bicyclist', 'motorcyclist',
    #              'moving-bicyclist', 'moving-motorcyclist'],  # riders are labeled as bikes in Audi dataset
    #     'person': ['person', 'moving-person'],
    #     'road': ['road', 'lane-marking'],
    #     'parking': ['parking'],
    #     'sidewalk': ['sidewalk'],
    #     'building': ['building'],
    #     'nature': ['vegetation', 'trunk', 'terrain'],
    #     'other-objects': ['fence', 'pole', 'traffic-sign', 'other-object'],
    # }

    # 6 class
    categories = {
        'vegetation_terrain': ['vegetation', 'trunk', 'terrain'],
        'building': ['building'],
        'road': ['road', 'lane-marking'],
        'object': ['fence', 'pole', 'traffic-sign', 'other-object'],
        'truck': ['truck', 'moving-truck'],
        'car': ['car', 'moving-car'],
    }

    # 9 class
    # categories = {
    # 'car': ['car', 'moving-car', 'truck', 'moving-truck'],
    # 'bike': ['bicycle', 'motorcycle', 'bicyclist', 'motorcyclist',
    #          'moving-bicyclist', 'moving-motorcyclist'],  # riders are labeled as bikes in Audi dataset
    # 'person': ['person', 'moving-person'],
    # 'road': ['road', 'lane-marking', 'parking'],
    # 'sidewalk': ['sidewalk'],
    # 'building': ['building'],
    # 'nature': ['vegetation', 'trunk', 'terrain'],
    # 'pole': ['pole'],
    # 'other-objects': ['fence', 'traffic-sign', 'other-object'],
    # }

    # zero-shot 6 classes
    # categories = {
    #     "motorcycle": ["motorcycle"],
    #     "truck": ["truck"],
    #     #"bicyclist": ["bicyclist"], # only 4 points in total dataset, we do not use it
    #     "traffic-sign": ["traffic-sign"],
    #     "car": ["car"],
    #     "terrain": ["terrain"],
    # }

    def __init__(self,
                 split,
                 preprocess_dir,
                 merge_classes=False
                 ):

        self.split = split
        self.preprocess_dir = preprocess_dir

        print("Initialize SemanticKITTI dataloader")

        assert isinstance(split, tuple)
        print('Load', split)
        self.data = []
        for curr_split in split:
            with open(osp.join(self.preprocess_dir, curr_split + '.pkl'), 'rb') as f:
                self.data.extend(pickle.load(f))

        if merge_classes:
            highest_id = list(self.id_to_class_name.keys())[-1]
            self.label_mapping = -100 * np.ones(highest_id + 2, dtype=int)
            for cat_idx, cat_list in enumerate(self.categories.values()):
                for class_name in cat_list:
                    self.label_mapping[self.class_name_to_id[class_name]] = cat_idx
            self.class_names = list(self.categories.keys())
        else:
            self.label_mapping = None

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class SemanticKITTI(SemanticKITTIBase):
    def __init__(self,
                 split,
                 preprocess_dir,
                 semantic_kitti_dir='',
                 merge_classes=False,
                 crop_size =tuple(),
                 ):
        super().__init__(split,
                         preprocess_dir,
                         merge_classes=merge_classes)

        self.semantic_kitti_dir = semantic_kitti_dir
        # self.resize = resize
        self.crop_size = crop_size

    def __getitem__(self, index):
        data_dict = self.data[index]
        num_classes = len(self.categories)
        pts_cam_coord = data_dict['pts_cam_coord'].numpy()
        seg_label = data_dict['seg_labels']
        if seg_label is not None:
            seg_label = seg_label.astype(np.int64)
        if self.label_mapping is not None and seg_label is not None:
            seg_label = self.label_mapping[seg_label]
        out_dict = {}
        points_img = data_dict['points_img']
        img_path = osp.join(self.semantic_kitti_dir, data_dict['camera_path'])
        image = Image.open(img_path)

        if self.crop_size:
            left = (image.size[0] - self.crop_size[0]) // 2
            right = left + self.crop_size[0]
            top = image.size[1] - self.crop_size[1]
            bottom = image.size[1]

            # discard points outside of crop
            keep_idx = points_img[:, 0] >= top
            keep_idx = np.logical_and(keep_idx, points_img[:, 0] < bottom)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
            keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)

            # crop image
            image = image.crop((left, top, right, bottom))
            points_img = points_img[keep_idx]
            points_img[:, 0] -= top
            points_img[:, 1] -= left

            # update point cloud
            pts_cam_coord = pts_cam_coord[keep_idx]
            if seg_label is not None:
                seg_label = seg_label[keep_idx]

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
