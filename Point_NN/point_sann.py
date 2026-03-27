# Non-Parametric Networks for 3D Point Cloud Part Segmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils

from .model_utils import *
import open3d as o3d


# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        # FPS
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long()
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)

        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)

        return lc_xyz, lc_x, knn_xyz, knn_x


# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):
        """
        Local Geometry Aggregation using Cross-Attention.
        :param lc_xyz: Coordinates of the sampled points, shape: (B, N, 3)
        :param lc_x: Features of the sampled points, shape: (B, N, C)
        :param knn_xyz: Coordinates of k nearest neighbors for each sampled point, shape: (B, N, K, 3)
        :param knn_x: Features of k nearest neighbors for each sampled point, shape: (B, N, K, C)
        :return: Aggregated features for each sampled point, shape: (B, C, N)
        """

        # get similarity scores
        lc_x_expanded = lc_x.unsqueeze(-2)  # (B, N, 1, C)
        sim_scores = F.cosine_similarity(lc_x_expanded, knn_x, dim=-1)  # (B, N, K)
        sim_weights = F.softmax(sim_scores, dim=-1).unsqueeze(-1)
        weighted_x = sim_weights * knn_x  # (B, N, K, C)

        # get distances weight
        distances = torch.norm(lc_xyz.unsqueeze(2) - knn_xyz, dim=-1, keepdim=True)  # Shape: (B, N, K, 1)
        # Normalize distances to get weights
        sigma = 0.5
        distance_weights = torch.exp(-distances ** 2 / (2 * sigma ** 2))
        distance_weights = distance_weights / torch.sum(distance_weights, dim=2, keepdim=True)  # Normalize to sum to 1

        aggregated_features = 0.25 * torch.sum(distance_weights * weighted_x, dim=2) + 0.75 * lc_x  # (B, N, C)

        aggregated_features = aggregated_features.permute(0, 2, 1)  # (B, C, N)

        return aggregated_features


# Raw-point Embedding
class Feature_Initial(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz):
        # xyz B,3,N
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.squeeze(0).permute(1, 0).cpu().numpy())  # N,3
        max_nn = 16
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max_nn))

        fpfh_feature = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamKNN(knn=max_nn)
        )
        fpfh_feature_np = fpfh_feature.data.copy()
        fpfh_feature = torch.from_numpy(fpfh_feature_np.T).to(xyz.device).unsqueeze(0).permute(0, 2, 1)  # shape B, C, N

        fpfh_feature_norm = (fpfh_feature - torch.min(fpfh_feature)) / (
                torch.max(fpfh_feature) - torch.min(fpfh_feature))
        xyz_norm = (xyz - torch.min(xyz)) / (torch.max(xyz) - torch.min(xyz))
        geometry_features = torch.cat([fpfh_feature_norm, xyz_norm], dim=1)

        return geometry_features


# Non-Parametric Encoder
class EncNP(nn.Module):
    def __init__(self, input_points, num_stages, k_neighbors):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages

        # Raw-point Embedding
        self.point_embed = Feature_Initial()

        self.FPS_kNN_list = nn.ModuleList()  # FPS, kNN
        self.LGA_list = nn.ModuleList()  # Local Geometry Aggregation

        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            group_num = group_num // 2
            k_neighbors = max(k_neighbors // 2, 2)
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.LGA_list.append(LGA())

    def forward(self, xyz, x):

        # Raw-point Embedding
        x = self.point_embed(x)

        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]  # [B, C, N]

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))
            # Local Geometry Aggregation
            x = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)

            xyz_list.append(xyz)
            x_list.append(x)

        return xyz_list, x_list


# Non-Parametric Decoder
class DecNP(nn.Module):
    def __init__(self, num_stages, de_neighbors):
        super().__init__()
        self.num_stages = num_stages
        self.de_neighbors = de_neighbors

    def propagate(self, xyz1, xyz2, x1, x2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            x1: input feature, [B, D', N]
            x2: input feature, [B, D'', S]
        Return:
            new_x: upsampled feature, [B, D''', N]
        """

        x2 = x2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_x = x2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :self.de_neighbors], idx[:, :, :self.de_neighbors]  # top-de_neighbors

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            weight = weight.view(B, N, self.de_neighbors, 1)

            interpolated_x = torch.sum(index_points(x2, idx) * weight, dim=2)

        if x1 is not None:
            x1 = x1.permute(0, 2, 1)
            new_x = torch.cat([x1, interpolated_x], dim=-1)
        else:
            new_x = interpolated_x

        new_x = new_x.permute(0, 2, 1)
        return new_x

    def forward(self, xyz_list, x_list):
        xyz_list.reverse()
        x_list.reverse()

        x = x_list[0]
        for i in range(self.num_stages):
            # Propagate point features to neighbors
            x = self.propagate(xyz_list[i + 1], xyz_list[i], x_list[i + 1], x)
        return x


# Non-Parametric Network
class Point_NN_Seg(nn.Module):
    def __init__(self, input_points=2048, num_stages=5, k_neighbors=128, de_neighbors=6):
        super().__init__()
        # Non-Parametric Encoder and Decoder
        self.EncNP = EncNP(input_points, num_stages, k_neighbors)
        self.DecNP = DecNP(num_stages, de_neighbors)

    def forward(self, x):
        # xyz: point coordinates B,N,3
        # x: point features B,C,N
        xyz = x.permute(0, 2, 1)

        # Non-Parametric Encoder
        xyz_list, x_list = self.EncNP(xyz, x)

        # Non-Parametric Decoder
        x = self.DecNP(xyz_list, x_list)
        return x
