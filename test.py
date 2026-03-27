from data.utils.evaluate import Evaluator
from common.utils.metric_logger import MetricLogger
from common.utils.logger import setup_logger
from common.utils.Binary_multi_view_clustering import binary_multi_view_clustering

import os
import os.path as osp
import argparse
import warnings
import time
import socket
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from scipy.optimize import linear_sum_assignment

torchvision.disable_beta_transforms_warning()
from torchvision import transforms
from data.build import build_dataloader
from PIL import Image

from Point_NN.point_sann import Point_NN_Seg

from DAM.metric_depth.depth_anything_v2.dpt import DepthAnythingV2


def preprocess_image(image, patch_size=14):
    width, height = image.size

    # Adjust the image size to ensure that the height and width are integer multiples of the patch size
    new_height = (height // patch_size) * patch_size
    new_width = (width // patch_size) * patch_size
    image = image.resize((new_width, new_height), Image.BICUBIC)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform(image).unsqueeze(0)


def infer_dino_feat(image, dino_model, img_indices, device='cpu'):
    to_pil_image = transforms.ToPILImage()
    image_pil = to_pil_image(image.permute(2, 0, 1))

    patch_size = 14
    image_width, image_height = image_pil.size
    num_patches_h = image_height // patch_size
    num_patches_w = image_width // patch_size

    dino_input = preprocess_image(image_pil, patch_size).to(device)
    features_dict = dino_model.forward_features(dino_input)
    dino_features = features_dict['x_norm_patchtokens']

    # Reshape feature maps
    dino_features = dino_features.reshape(-1, num_patches_h, num_patches_w, dino_features.shape[-1])
    dino_features = dino_features.permute(0, 3, 1, 2)

    # Use bicubic interpolation to map feature maps to pixel by pixel feature maps
    image_features = torch.nn.functional.interpolate(
        dino_features, size=(image_height, image_width), mode='bicubic', align_corners=False
    )

    image_features = image_features.squeeze(0).permute(1, 2, 0)[img_indices[:, 0], img_indices[:, 1]].to(
        torch.double)

    image_features = (image_features - torch.min(image_features)) / (
            torch.max(image_features) - torch.min(image_features))

    return image_features


def parse_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument('--ckpt', type=str, help='path to checkpoint file of the 3D model', default='')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def test(cfg, logger):
    # build dataset
    logger.propagate = False
    logger.info('Start testing')
    dataloader = build_dataloader(cfg, mode='test', domain='target')
    class_names = dataloader.dataset.class_names
    evaluator_3d = Evaluator(class_names)
    val_metric_logger = MetricLogger(delimiter='  ')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load dinov2 model
    logger.info('Loading dinov2')
    dino_model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14")
    dino_model.eval()
    dino_model.to(device)

    # load Depth anything model
    logger.info('Loading DAM')
    dam_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    encoder = 'vitl'  # or 'vits', 'vitb'
    dataset = 'vkitti'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 80  # 20 for indoor model, 80 for outdoor model

    dam = DepthAnythingV2(**{**dam_configs[encoder], 'max_depth': max_depth})
    dam.load_state_dict(
        torch.load(f'DAM/metric_depth/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth',
                   map_location='cpu'))
    dam.to(device).eval()

    with torch.no_grad():
        logger.info('Starting evaluation')
        end = time.time()
        # evaluation
        for iteration, data_batch in enumerate(dataloader):
            data_time = time.time() - end

            num_classes = data_batch['num_classes']

            img_indices = torch.from_numpy(data_batch['img_indices'][0]).to(device)
            # while the points are limited
            if len(img_indices) <= 32:
                continue

            # calculate image features (2d)
            image = data_batch['img'][0].to(device) * 255
            image_features = infer_dino_feat(image, dino_model, img_indices, device)

            # calculate depth map features
            depth = dam.infer_image(image.cpu().numpy())
            depth = ((depth - depth.min()) / (depth.max() - depth.min()) * 255.0).to(torch.uint8).unsqueeze(-1).repeat(
                1, 1, 3)
            depth_features = infer_dino_feat(depth, dino_model, img_indices, device)

            # calculate geometry features (3d)
            pc = data_batch['pc'][0].unsqueeze(0).permute(0, 2, 1).float().to(device)
            input_points = torch.tensor(pc.shape[2])
            k_neighbors = input_points // 16
            num_stages = (torch.floor(torch.log2(input_points)) - torch.tensor([3])).to(torch.int)
            point_sann = Point_NN_Seg(input_points=input_points, num_stages=num_stages, k_neighbors=k_neighbors).to(
                device).eval()

            geometry_features = point_sann(pc).permute(0, 2, 1).squeeze(0).to(torch.double)
            geometry_features = (geometry_features - torch.min(geometry_features)) / (
                    torch.max(geometry_features) - torch.min(geometry_features))

            ## Cluster ##
            seg_label = data_batch['seg_label'].squeeze(0)
            true_labels = seg_label.numpy()
            MV_feature = [image_features, depth_features, geometry_features]

            ### BMCV setting
            ## UDA setting ##
            anchor_dim = 500
            L = 32
            r = 5
            beta = 0.003
            ## UDA setting ##

            ## ZSL setting ##
            # anchor_dim = 30
            # L = 8
            # r = 2
            # beta = 0.03
            ## ZSL setting ##

            cluster_indices = binary_multi_view_clustering(X=MV_feature, MaxIter=300, L=L, anchor_dim=anchor_dim, r=r,
                                                           beta=beta, init_view=cfg.BMVC_INIT_VIEW,
                                                           n_cluster=num_classes, device=device)

            # Using the Hungarian algorithm to find the optimal label mapping
            confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
            for i in range(cluster_indices.shape[0]):
                confusion_matrix[true_labels[i], cluster_indices[i]] += 1
            row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
            mapping = {col: row for row, col in zip(row_ind, col_ind)}
            pred_label = np.vectorize(mapping.get)(cluster_indices)

            evaluator_3d.update(pred_label, seg_label)
            pred_seg_logit = F.one_hot(torch.from_numpy(pred_label).to(device), num_classes).to(
                torch.float32)
            seg_loss_3d = F.cross_entropy(pred_seg_logit, seg_label.to(device))
            if seg_loss_3d is not None:
                val_metric_logger.update(seg_loss_3d=seg_loss_3d)

            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()
            # log
            cur_iter = iteration + 1
            if cur_iter == 1 or (cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.VAL.LOG_PERIOD == 0):
                logger.info(
                    val_metric_logger.delimiter.join(
                        [
                            'iter: {iter}/{total_iter}',
                            '{meters}'
                        ]
                    ).format(
                        iter=cur_iter,
                        total_iter=len(dataloader),
                        meters=str(val_metric_logger)
                    )
                )

        eval_list = []
        if evaluator_3d is not None:
            val_metric_logger.update(seg_iou_3d=evaluator_3d.overall_iou)
            eval_list.append(('3D', evaluator_3d))
        for modality, evaluator in eval_list:
            logger.info('{} overall accuracy: {:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU: {:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
            logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.format(modality, evaluator.print_table()))


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from common.config import purge_cfg
    from configs.config import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs/', ''))
        if not osp.isdir(output_dir):
            warnings.warn('Make a new directory: {}'.format(output_dir))
            os.makedirs(output_dir)

    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)
    logger = setup_logger('Training_free', output_dir, comment='test.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))

    logger.info('Loaded configuration file {:s}'.format(args.config_file))

    test(cfg, logger)


if __name__ == '__main__':
    main()
