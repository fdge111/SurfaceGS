#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.sh_utils import RGB2SH, IDFT
from simple_knn._C import distCUDA2
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import quaternion_to_rotation_matrix
import torch.nn.functional as F
import logging
from scene.deform_model import DeformModel
from sklearn.cluster import KMeans

try:
    import tensorly as tl
    tl.set_backend('pytorch')
except ImportError:
    tl = None


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.normal_activation = lambda x: torch.nn.functional.normalize(x, dim=-1, eps=1e-3)

        self.BezierCoeff = self.get_bezier_coefficient()
        self.BezierDerivativeCoeff = self.get_d_bezier_coefficient()

        # ------------------------------------------
        # addition for Bezier/BSpline surface
        self.SurfCoeff = self.get_bspline_surface_coefficient()
        # ------------------------------------------

    def __init__(self, args):
        self.active_sh_degree = 0
        self.fourier_dim = args.fourier_dim
        self.num_control_points = args.order + 1  # order为3时表示3阶贝塞尔曲线，对应4个控制点
        # ------------------------------------------
        # addition for BSpline surface
        self.num_surface_u_control_points = args.num_cp_u
        self.num_surface_v_control_points = args.num_cp_v
        self.surface_u_order = args.surface_u_order
        self.surface_v_order = args.surface_v_order
        self.num_patches = args.num_patches
        use_seg_hetero_dynamics = getattr(args, "use_seg_hetero_dynamics", False)
        if isinstance(use_seg_hetero_dynamics, str):
            use_seg_hetero_dynamics = use_seg_hetero_dynamics.lower() not in {"0", "false", "no", "off"}
        self.use_seg_hetero_dynamics = bool(use_seg_hetero_dynamics)
        self.low_rank_decom_order = args.low_rank_decom_order
        # ------------------------------------------
        self.max_sh_degree = args.sh_degree
        self._control_points = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._group = torch.empty(0)
        # ------------------------------------------
        # addition for surface
        self._cp_group = torch.empty(0)
        self._gaussian_uv = torch.empty(0)
        self._cp_idx_to_ids = torch.empty(0)
        self._point_to_cp_idx = torch.empty(0)

        self._core_tensors = torch.empty(0)
        self._factor_U = torch.empty(0)
        self._factor_V = torch.empty(0)
        self._factor_T = torch.empty(0)

        self.use_low_rank = args.use_low_rank
        # UV coordinates and shared low-rank factors are stable after initialization
        # in the default fast path. Both can still be trained explicitly for ablations.
        self.optimize_surface_uv = getattr(args, "optimize_surface_uv", True)
        self.optimize_low_rank_factors = getattr(args, "optimize_low_rank_factors", False)
        self.optimize_static_control_points = getattr(args, "optimize_static_control_points", True)
        # ------------------------------------------

        self.dynamic_mode = args.dynamic_mode  # Bezier curve, PVG sin, DeformableGS MLP
        self.deform = DeformModel(False)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.time_duration = args.time_duration
        self.frame_interval_cnt = args.end_time - args.start_time
        self.no_time_split = args.no_time_split
        self.contract = args.contract
        self.t_init = args.t_init
        self.big_point_threshold = args.big_point_threshold

        self.opa_init = args.opa_init

        # self.velocity_decay = args.velocity_decay # 1.0
        self.random_init_point = args.random_init_point

        self.binomial_coefs = torch.tensor(
            [math.comb(self.num_control_points - 1, k) for k in range(self.num_control_points)],
            dtype=torch.float32,
        ).cuda()  # 二项式系数C(n, k) = n!/(k!(n-k)!)
        # ------------------------------------------
        # addition for Bezier surface
        # self.surface_u_binomial_coefs = torch.tensor(
        #     [math.comb(self.num_surface_u_control_points - 1, k) for k in range(self.num_surface_u_control_points)],
        #     dtype=torch.float32,
        # ).cuda()
        # self.surface_v_binomial_coefs = torch.tensor(
        #     [math.comb(self.num_surface_v_control_points - 1, k) for k in range(self.num_surface_v_control_points)],
        #     dtype=torch.float32,
        # ).cuda()

        # 生成 u 和 v 方向的节点向量
        self.knots_u = self.generate_knot_vector(self.num_surface_u_control_points, self.surface_u_order)
        self.knots_v = self.generate_knot_vector(self.num_surface_v_control_points, self.surface_v_order)
        # ------------------------------------------

        self.group_num = 0
        self.g_id = 0
        self.world_bezier_time_map = dict()
        self.trajectory_dict = dict()
        self.trajectory_cp_tensor = None
        self.cp_recorded_timestamp_2_bezier_t = None
        self.exist_range = dict()

        # ------------------------------------------
        # addition for Bezier surface
        self.grid_resolution = args.grid_resolution
        # 创建一个 [res, res] 的 uv 网格，坐标范围 [0, 1]
        u_coords = torch.linspace(0, 1, self.grid_resolution).cuda()
        v_coords = torch.linspace(0, 1, self.grid_resolution).cuda()
        grid_v, grid_u = torch.meshgrid(v_coords, u_coords, indexing='ij')
        self.grid_uv = torch.stack([grid_u, grid_v], dim=-1).view(-1, 2)  # [res*res, 2]
        # ------------------------------------------

        self.setup_functions()

    # to be modified
    def capture(self):
        return (
            self.active_sh_degree,
            self._control_points,
            self._core_tensors,
            self._factor_U,
            self._factor_V,
            self._factor_T,
            self._gaussian_uv,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._group,
            self._cp_group,
            self._cp_idx_to_ids,
            self._point_to_cp_idx,
            self._cp_remap_table,
            self._dynamic_cp_mask,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args=None):
        (
            self.active_sh_degree,
            self._control_points,
            self._core_tensors,
            self._factor_U,
            self._factor_V,
            self._factor_T,
            self._gaussian_uv,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._group,
            self._cp_group,
            self._cp_idx_to_ids,
            self._point_to_cp_idx,
            self._cp_remap_table,
            self._dynamic_cp_mask,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        if not self.use_low_rank:
            empty = torch.empty(
                0, dtype=self._control_points.dtype, device=self._control_points.device
            )
            self._core_tensors = empty
            self._factor_U = empty
            self._factor_V = empty
            self._factor_T = empty
        self.setup_functions()
        self._rebuild_group_point_indices()
        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    # ------------------------------------------
    # change for Bezier surface
    @property
    def get_xyz(self):
        # return self._control_points[:, 0, :]
        return self._control_points[self._point_to_cp_idx][:, 0, 0, 0, :] # 现在是object-surface-point，而不是object-point，因此需要加上额外索引_point_to_cp_idx
    # ------------------------------------------

    def get_feats(self, time=None):
        if time is None:
            return self.get_features
        else:
            return self.get_features4d(time)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    def get_features4d(self, time):  # 4DSH
        all_features = []
        for idx in range(self.group_num + 1):
            if idx == 0:
                normalized_time = time
            else:
                normalized_time = (time - self.exist_range[idx][0]) / (self.exist_range[idx][1] - self.exist_range[idx][0])
            
            mask = self.get_group == idx
            idft_base = IDFT(normalized_time, self.fourier_dim)[0].cuda()
            features_dc = self._features_dc[mask] # [N, C, 3]
            features_dc = torch.sum(features_dc * idft_base[..., None], dim=1, keepdim=True) # [N, 1, 3]
            features_rest = self._features_rest[mask] # [N, sh, 3]
            features = torch.cat([features_dc, features_rest], dim=1) # [N, (sh + 1) * C, 3]
            all_features.append(features)
        return torch.cat(all_features, dim=0)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    # @property
    # def get_control_points(self):
    #     return self._control_points

    @property
    def get_gaussian_uv(self):
        return self._gaussian_uv

    @property
    def get_group(self):
        return self._group

    @property
    def get_cp_group(self):
        return self._cp_group

    @torch.no_grad()
    def _rebuild_group_point_indices(self):
        """Rebuild point-level lookup tables after initialization or densification."""
        group = self.get_group
        self._group_point_indices = [
            torch.where(group == idx)[0] for idx in range(self.group_num + 1)
        ]
        self._dynamic_point_indices = torch.where(group > 0)[0]
        self._dynamic_point_groups = group[self._dynamic_point_indices].long()
        self._dynamic_point_cp_indices = self._point_to_cp_idx[self._dynamic_point_indices]
        self._dynamic_surface_basis_cache = None

    def _get_dynamic_eval_data(self, world_time):
        """Collect all active dynamic points so surface evaluation needs one GPU call."""
        if (
            not hasattr(self, "_dynamic_point_indices")
            or len(self._group_point_indices) != self.group_num + 1
        ):
            self._rebuild_group_point_indices()
        if self.group_num <= 0 or self._dynamic_point_indices.numel() == 0:
            return None

        device = self._gaussian_uv.device
        dtype = self._control_points.dtype
        world_time_tensor = torch.as_tensor([[world_time]], device=device, dtype=dtype)
        world_basis = self.BezierCoeff(world_time_tensor).to(dtype=dtype, device=device)
        bezier_t_all = torch.matmul(
            world_basis.unsqueeze(0),
            self.cp_recorded_timestamp_2_bezier_t[:self.group_num].to(
                dtype=dtype, device=device
            ),
        ).squeeze(1)

        active_groups = [
            self.exist_range[idx][0] <= world_time <= self.exist_range[idx][1]
            for idx in range(1, self.group_num + 1)
        ]
        if not any(active_groups):
            return None

        dynamic_selection = None
        point_indices = self._dynamic_point_indices
        point_groups = self._dynamic_point_groups
        cp_indices = self._dynamic_point_cp_indices
        if not all(active_groups):
            active_group_mask = torch.tensor(active_groups, dtype=torch.bool, device=device)
            dynamic_selection = active_group_mask[point_groups - 1]
            point_indices = point_indices[dynamic_selection]
            point_groups = point_groups[dynamic_selection]
            cp_indices = cp_indices[dynamic_selection]

        return point_indices, point_groups, cp_indices, bezier_t_all, dynamic_selection

    def _get_dynamic_surface_basis(self, dynamic_selection=None):
        """Reuse UV-dependent B-spline basis values when UV optimization is disabled."""
        point_indices = self._dynamic_point_indices
        if self._gaussian_uv.requires_grad:
            if dynamic_selection is not None:
                point_indices = point_indices[dynamic_selection]
            uvs = self._gaussian_uv[point_indices]
            return self.SurfCoeff(uvs[:, :1], uvs[:, 1:])

        if self._dynamic_surface_basis_cache is None:
            uvs = self._gaussian_uv[point_indices]
            basis_u, basis_v = self.SurfCoeff(uvs[:, :1], uvs[:, 1:])
            self._dynamic_surface_basis_cache = basis_u.detach(), basis_v.detach()

        if dynamic_selection is None:
            return self._dynamic_surface_basis_cache
        return tuple(basis[dynamic_selection] for basis in self._dynamic_surface_basis_cache)

    def _get_dynamic_tensors(self, cp_indices):
        if not self.use_low_rank:
            return self._control_points[cp_indices]
        if self._core_tensors.shape[0] == self._control_points.shape[0]:
            return self._core_tensors[cp_indices]
        return self._core_tensors[self._cp_remap_table[cp_indices]]

    def get_xyz_with_surface_offset(self, world_time):
        """Evaluate static positions and all active dynamic surfaces in one batch."""
        if self.dynamic_mode != "Bezier":
            raise NotImplementedError("Surface offsets currently support Bezier mode only.")

        device = self._gaussian_uv.device
        dtype = self._control_points.dtype
        num_points = self._gaussian_uv.shape[0]
        xyz = torch.empty(num_points, 3, device=device, dtype=dtype)
        xyz_offset = torch.zeros_like(xyz)
        valid_mask = torch.zeros(num_points, dtype=torch.bool, device=device)

        if not hasattr(self, "_group_point_indices"):
            self._rebuild_group_point_indices()
        static_point_indices = self._group_point_indices[0]
        if static_point_indices.numel() > 0:
            static_cp_indices = self._point_to_cp_idx[static_point_indices]
            xyz[static_point_indices] = self._control_points[static_cp_indices, 0, 0, 0, :]
            valid_mask[static_point_indices] = True

        dynamic_data = self._get_dynamic_eval_data(world_time)
        if dynamic_data is None:
            return xyz, xyz_offset, valid_mask

        point_indices, point_groups, cp_indices, bezier_t_all, selection = dynamic_data
        point_bezier_t = bezier_t_all[point_groups - 1]
        surface_basis = self._get_dynamic_surface_basis(selection)
        tensors_for_points = self._get_dynamic_tensors(cp_indices)

        if self.use_low_rank:
            dynamic_offsets = self.predict_xyz_surface_low_rank(
                point_bezier_t, tensors_for_points, surface_basis=surface_basis
            )
        else:
            dynamic_offsets = self.predict_xyz_surface(
                point_bezier_t, tensors_for_points, surface_basis=surface_basis
            )

        trajectory_basis = self.BezierCoeff(bezier_t_all).to(dtype=dtype, device=device)
        trajectory_centers = torch.sum(
            trajectory_basis.unsqueeze(-1)
            * self.trajectory_cp_tensor[:self.group_num].to(dtype=dtype, device=device),
            dim=1,
        )
        xyz_offset[point_indices] = dynamic_offsets
        xyz[point_indices] = dynamic_offsets + trajectory_centers[point_groups - 1]
        valid_mask[point_indices] = True
        return xyz, xyz_offset, valid_mask

    def predict_xyz_surface(
        self, t, batched_control_points, batched_uvs=None, surface_basis=None
    ):
        """Evaluate full-rank surfaces with a fixed, memory-efficient contraction path."""
        basis_t = self.BezierCoeff(t).to(
            dtype=batched_control_points.dtype, device=batched_control_points.device
        )
        if basis_t.shape[0] == 1 and batched_control_points.shape[0] != 1:
            basis_t = basis_t.expand(batched_control_points.shape[0], -1)
        if surface_basis is None:
            surface_basis = self.SurfCoeff(batched_uvs[:, :1], batched_uvs[:, 1:])
        basis_u, basis_v = surface_basis

        control_points_at_t = torch.einsum("nuvtc,nt->nuvc", batched_control_points, basis_t)
        control_points_at_u = torch.einsum("nuvc,nu->nvc", control_points_at_t, basis_u)
        return torch.einsum("nvc,nv->nc", control_points_at_u, basis_v)

    def predict_xyz_surface_low_rank(
        self, t, core_tensor, batched_uvs=None, surface_basis=None
    ):
        """Evaluate Tucker-compressed surfaces for one or many time values."""
        basis_t = self.BezierCoeff(t).to(dtype=core_tensor.dtype, device=core_tensor.device)
        if basis_t.shape[0] == 1 and core_tensor.shape[0] != 1:
            basis_t = basis_t.expand(core_tensor.shape[0], -1)
        if surface_basis is None:
            surface_basis = self.SurfCoeff(batched_uvs[:, :1], batched_uvs[:, 1:])
        basis_u, basis_v = surface_basis
        basis_u = basis_u @ self._factor_U
        basis_v = basis_v @ self._factor_V
        basis_t = basis_t @ self._factor_T
        return torch.einsum("nijkc,nk,ni,nj->nc", core_tensor, basis_t, basis_u, basis_v)

    def get_inst_surface_velocity(self, world_time):
        """Evaluate all active dynamic surface derivatives in one batch."""
        device = self._gaussian_uv.device
        dtype = self._control_points.dtype
        velocity = torch.zeros(self._gaussian_uv.shape[0], 3, device=device, dtype=dtype)
        valid_mask = torch.zeros(self._gaussian_uv.shape[0], dtype=torch.bool, device=device)

        if not hasattr(self, "_group_point_indices"):
            self._rebuild_group_point_indices()
        valid_mask[self._group_point_indices[0]] = True
        dynamic_data = self._get_dynamic_eval_data(world_time)
        if dynamic_data is None:
            return velocity, valid_mask

        point_indices, point_groups, cp_indices, bezier_t_all, selection = dynamic_data
        point_bezier_t = bezier_t_all[point_groups - 1]
        surface_basis = self._get_dynamic_surface_basis(selection)
        tensors_for_points = self._get_dynamic_tensors(cp_indices)
        if self.use_low_rank:
            dynamic_velocity = self.predict_xyz_surface_derivative_low_rank(
                point_bezier_t, tensors_for_points, surface_basis=surface_basis
            )
        else:
            dynamic_velocity = self.predict_xyz_surface_derivative(
                point_bezier_t, tensors_for_points, surface_basis=surface_basis
            )

        velocity[point_indices] = dynamic_velocity
        valid_mask[point_indices] = True
        return velocity, valid_mask

    def predict_xyz_surface_derivative(
        self, t, batched_control_points, batched_uvs=None, surface_basis=None
    ):
        """Evaluate full-rank temporal surface derivatives."""
        basis_t = self.BezierDerivativeCoeff(t).to(
            dtype=batched_control_points.dtype, device=batched_control_points.device
        ).squeeze(1)
        if basis_t.shape[0] == 1 and batched_control_points.shape[0] != 1:
            basis_t = basis_t.expand(batched_control_points.shape[0], -1)
        if surface_basis is None:
            surface_basis = self.SurfCoeff(batched_uvs[:, :1], batched_uvs[:, 1:])
        basis_u, basis_v = surface_basis

        control_points_at_t = torch.einsum("nuvtc,nt->nuvc", batched_control_points, basis_t)
        control_points_at_u = torch.einsum("nuvc,nu->nvc", control_points_at_t, basis_u)
        return torch.einsum("nvc,nv->nc", control_points_at_u, basis_v)

    def predict_xyz_surface_derivative_low_rank(
        self, t, core_tensor, batched_uvs=None, surface_basis=None
    ):
        """Evaluate Tucker-compressed temporal surface derivatives."""
        basis_t = self.BezierDerivativeCoeff(t).to(
            dtype=core_tensor.dtype, device=core_tensor.device
        ).squeeze(1)
        if basis_t.shape[0] == 1 and core_tensor.shape[0] != 1:
            basis_t = basis_t.expand(core_tensor.shape[0], -1)
        if surface_basis is None:
            surface_basis = self.SurfCoeff(batched_uvs[:, :1], batched_uvs[:, 1:])
        basis_u, basis_v = surface_basis
        basis_u = basis_u @ self._factor_U
        basis_v = basis_v @ self._factor_V
        basis_t = basis_t @ self._factor_T
        return torch.einsum("nijkc,nk,ni,nj->nc", core_tensor, basis_t, basis_u, basis_v)

    def generate_knot_vector(self, num_control_points: int, degree: int):
        """
        生成节点向量。

        参数:
            num_control_points (int): 控制点的数量 (n+1 或 m+1)。
            degree (int): B样条的阶数 (p 或 q)。
            device (str): "cpu" 或 "cuda"。

        返回:
            torch.Tensor: 生成的节点向量。
        """
        # 节点向量的长度为 num_control_points + degree + 1
        num_knots = num_control_points + degree + 1

        # 创建一个空节点向量
        knots = torch.zeros(num_knots).cuda()

        # 内部节点是均匀分布的
        knots[degree: num_knots - degree] = torch.linspace(0, 1, num_control_points - degree + 1).cuda()

        # 向量的末端填充1（钳位）
        knots[num_knots - degree:] = 1

        return knots


    def get_rotation_matrix(self):
        return quaternion_to_rotation_matrix(self.get_rotation)

    @property
    def get_max_sh_channels(self):
        return (self.max_sh_degree + 1) ** 2

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # ============================ 新增辅助函数 开始 ============================
    def _initial_patch_assignment_kmeans(self, points_xyz, num_patches):
        """
        使用 K-Means 对点云进行初始的面片划分。
        Args:
            points_xyz (torch.Tensor): 物体的平均点云位置, 形状 [N, 3]。
            num_patches (int): 希望划分的面片数量。
        Returns:
            torch.Tensor: 每个点所属的面片ID, 形状 [N]。
        """
        print(f"进行 K-Means 聚类以划分 {num_patches} 个面片...")
        kmeans = KMeans(n_clusters=num_patches, random_state=0, n_init='auto')
        points_np = points_xyz.detach().cpu().numpy()
        patch_ids = kmeans.fit_predict(points_np)
        print("K-Means 聚类完成。")
        return torch.from_numpy(patch_ids).long().to(points_xyz.device)

    def _run_kmeans(self, features: torch.Tensor, num_clusters: int) -> torch.Tensor:
        """
        对点级特征执行稳健 K-Means，并处理聚类数超过样本数的情况。
        """
        num_samples = features.shape[0]
        if num_samples == 0:
            return torch.empty((0,), dtype=torch.long, device=features.device)

        num_clusters = int(max(1, min(num_clusters, num_samples)))
        if num_clusters == 1:
            return torch.zeros((num_samples,), dtype=torch.long, device=features.device)

        features_np = torch.nan_to_num(
            features.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
        ).cpu().numpy()
        try:
            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
            labels_np = kmeans.fit_predict(features_np)
        except (TypeError, ValueError):
            # sklearn < 1.2 does not support n_init='auto'.
            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
            labels_np = kmeans.fit_predict(features_np)
        return torch.from_numpy(labels_np).long().to(features.device)

    def _normalize_feature_columns(self, features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        对聚类特征逐列标准化，避免空间尺度掩盖运动差异。
        """
        if features.numel() == 0:
            return features
        features = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
        return (features - mean) / std

    def _load_point_segmentation_value(self, value):
        """
        加载预处理阶段生成的点级标签。SAM 等 2D 模型的 mask 需要先投影到
        当前对象的 3D 点，再以数组或 .npy/.npz 路径写入 ply_dict。
        """
        if not isinstance(value, (str, os.PathLike)):
            return value

        labels_path = os.fspath(value)
        if not os.path.isfile(labels_path):
            print(f"警告：点级分割文件不存在，已忽略：{labels_path}")
            return None

        try:
            loaded = np.load(labels_path, allow_pickle=True)
        except (OSError, ValueError) as exc:
            print(f"警告：无法加载点级分割文件 {labels_path}：{exc}")
            return None

        if isinstance(loaded, np.lib.npyio.NpzFile):
            preferred_keys = [
                'labels', 'semantic_labels', 'part_labels', 'component_labels',
                'segment_ids', 'segmentation', 'masks',
            ]
            selected_key = next((key for key in preferred_keys if key in loaded.files), None)
            if selected_key is None and len(loaded.files) == 1:
                selected_key = loaded.files[0]
            if selected_key is None:
                print(f"警告：分割文件 {labels_path} 中无法确定标签字段，已忽略。")
                loaded.close()
                return None
            labels = loaded[selected_key]
            loaded.close()
            return labels
        return loaded

    def _labels_to_point_tensor(self, labels, num_points: int, device: torch.device):
        """
        将点标签或 SAM mask 矩阵转换为 [N] long Tensor。
        支持整数标签、字符串标签、[N,K] soft mask 与 [K,N] mask。
        """
        labels = self._load_point_segmentation_value(labels)
        if labels is None:
            return None

        if isinstance(labels, torch.Tensor):
            labels_t = labels.detach()
        else:
            labels_np = np.asarray(labels)
            if labels_np.dtype.kind in {'U', 'S', 'O'}:
                flat_labels = labels_np.reshape(-1)
                if flat_labels.shape[0] != num_points:
                    return None
                _, inverse = np.unique(flat_labels.astype(str), return_inverse=True)
                labels_t = torch.from_numpy(inverse).long()
            else:
                labels_t = torch.from_numpy(labels_np)

        if labels_t.ndim == 0:
            return None

        labels_t = labels_t.to(device)
        if labels_t.ndim == 1:
            if labels_t.numel() != num_points:
                return None
            return labels_t.long()

        if labels_t.ndim == 2:
            if labels_t.shape[0] == num_points and labels_t.shape[1] == 1:
                return labels_t[:, 0].long()
            if labels_t.shape[0] == num_points and labels_t.shape[1] > 1:
                max_scores, point_labels = torch.max(labels_t.float(), dim=1)
                point_labels[max_scores <= 0] = -1
                return point_labels.long()
            if labels_t.shape[1] == num_points and labels_t.shape[0] > 1:
                max_scores, point_labels = torch.max(labels_t.float(), dim=0)
                point_labels[max_scores <= 0] = -1
                return point_labels.long()
            if labels_t.numel() == num_points:
                return labels_t.reshape(-1).long()

        return None

    def _extract_component_labels_from_ply_entry(
        self,
        obj_record: dict,
        num_points: int,
        device: torch.device,
    ):
        """
        从动态对象记录中读取由 SAM 等模型预生成并投影到 3D 点的分割标签。
        """
        candidate_keys = [
            'sam_labels', 'sam_part_labels', 'sam_component_labels', 'sam_segmentation',
            'sam_masks', 'sam_labels_path', 'sam_segmentation_path',
            'semantic_labels', 'semantic_label', 'semantics', 'semantic',
            'semantic_labels_path', 'part_labels', 'part_label', 'part_ids', 'part_id',
            'part_labels_path', 'component_labels', 'component_label', 'component_ids',
            'component_id', 'object_part_labels', 'point_part_labels',
            'motion_labels', 'motion_label', 'motion_segment_ids', 'motion_segmentation',
            'seg_labels', 'seg_label', 'segment_ids', 'seg_ids', 'segmentation',
            'segmentation_path', 'plane_labels', 'plane_label', 'plane_ids', 'plane_id',
        ]

        for key in candidate_keys:
            if key not in obj_record:
                continue
            labels_t = self._labels_to_point_tensor(obj_record[key], num_points, device)
            if labels_t is None:
                print(f"警告：检测到 {key}，但其形状无法与点数 {num_points} 对齐，已忽略。")
                continue

            valid_mask = labels_t >= 0
            if not valid_mask.any():
                print(f"警告：检测到 {key}，但没有有效的点级标签，已忽略。")
                continue

            compact = torch.full_like(labels_t, -1)
            _, compact_valid = torch.unique(
                labels_t[valid_mask], sorted=True, return_inverse=True
            )
            compact[valid_mask] = compact_valid
            print(f"检测到异质动态分割字段 '{key}'，包含 {torch.unique(compact_valid).numel()} 个部件/语义区域。")
            return compact.long()

        return None

    def _build_motion_aware_features(
        self,
        xyz_offset: torch.Tensor,
        mean_shape_xyz: torch.Tensor,
    ) -> torch.Tensor:
        """
        构造点级空间与运动特征，使同一平面上的不同运动模式可以被分开。
        """
        num_points = mean_shape_xyz.shape[0]
        device = mean_shape_xyz.device
        dtype = mean_shape_xyz.dtype
        if num_points == 0:
            return torch.empty((0, 1), dtype=dtype, device=device)

        centered_xyz = mean_shape_xyz - mean_shape_xyz.mean(dim=0, keepdim=True)
        if xyz_offset.ndim == 3 and xyz_offset.shape[1] > 1:
            displacement = xyz_offset[:, -1, :] - xyz_offset[:, 0, :]
            velocity = xyz_offset[:, 1:, :] - xyz_offset[:, :-1, :]
            speed = torch.linalg.norm(velocity, dim=-1)
            mean_velocity = velocity.mean(dim=1)
            speed_mean = speed.mean(dim=1, keepdim=True)
            speed_std = speed.std(dim=1, keepdim=True, unbiased=False)
            deformation_amp = torch.linalg.norm(
                xyz_offset - xyz_offset.mean(dim=1, keepdim=True),
                dim=-1,
            ).mean(dim=1, keepdim=True)
        else:
            displacement = torch.zeros((num_points, 3), dtype=dtype, device=device)
            mean_velocity = torch.zeros((num_points, 3), dtype=dtype, device=device)
            speed_mean = torch.zeros((num_points, 1), dtype=dtype, device=device)
            speed_std = torch.zeros((num_points, 1), dtype=dtype, device=device)
            deformation_amp = torch.zeros((num_points, 1), dtype=dtype, device=device)

        radial = torch.linalg.norm(centered_xyz, dim=-1, keepdim=True)
        features = torch.cat([
            0.5 * centered_xyz,
            1.5 * displacement,
            1.5 * mean_velocity,
            speed_mean,
            speed_std,
            deformation_amp,
            0.5 * radial,
        ], dim=-1)
        return self._normalize_feature_columns(features)

    def _fill_unassigned_component_labels(
        self,
        component_ids: torch.Tensor,
        features: torch.Tensor,
    ):
        """
        将 SAM 未覆盖的点分配给运动特征中最近的已知语义组件。
        """
        valid_mask = component_ids >= 0
        if valid_mask.all():
            return component_ids
        if not valid_mask.any():
            return None

        valid_component_ids = torch.unique(component_ids[valid_mask], sorted=True)
        centers = torch.stack([
            features[component_ids == component_id].mean(dim=0)
            for component_id in valid_component_ids
        ], dim=0)
        distance = torch.cdist(features[~valid_mask], centers)
        component_ids = component_ids.clone()
        component_ids[~valid_mask] = valid_component_ids[torch.argmin(distance, dim=1)]
        return component_ids

    def _allocate_patch_budget_per_component(
        self,
        component_ids: torch.Tensor,
        target_num_patches: int,
    ):
        """
        保证每个异质动态组件至少获得一个 patch，再把剩余 patch 分给较大的组件。
        """
        unique_components = torch.unique(component_ids, sorted=True)
        num_points = component_ids.shape[0]
        target_total = min(num_points, max(int(target_num_patches), int(unique_components.numel())))
        budgets = {int(cid.item()): 1 for cid in unique_components}
        component_sizes = {
            int(cid.item()): int((component_ids == cid).sum().item())
            for cid in unique_components
        }

        remaining = target_total - len(budgets)
        while remaining > 0:
            candidates = [
                cid for cid, count in component_sizes.items()
                if budgets[cid] < count
            ]
            if len(candidates) == 0:
                break
            best_component = max(
                candidates,
                key=lambda cid: component_sizes[cid] / float(budgets[cid] + 1),
            )
            budgets[best_component] += 1
            remaining -= 1
        return budgets

    def _segmentation_for_heterogeneous_dynamics(
        self,
        xyz_offset: torch.Tensor,
        mean_shape_xyz: torch.Tensor,
        obj_record: dict,
        num_patches: int,
    ):
        """
        Segmentation for heterogeneous dynamics.

        优先按预处理提供的 SAM/语义部件标签分开 wheel、chassis 等独立运动区域；
        没有显式标签时，使用点的 offset、速度和形变幅度执行 motion-aware K-Means。
        """
        num_points = mean_shape_xyz.shape[0]
        device = mean_shape_xyz.device
        if not self.use_seg_hetero_dynamics:
            print("已关闭 Segmentation for heterogeneous dynamics，当前动态对象仅使用基于曲率的曲面划分。")
            zeros = torch.zeros((num_points,), dtype=torch.long, device=device)
            return zeros, torch.zeros((1,), dtype=torch.long, device=device)

        target_patches = int(max(1, min(num_patches, num_points)))
        motion_features = self._build_motion_aware_features(xyz_offset, mean_shape_xyz)
        component_ids = self._extract_component_labels_from_ply_entry(
            obj_record,
            num_points,
            device,
        )
        if component_ids is not None:
            component_ids = self._fill_unassigned_component_labels(
                component_ids,
                motion_features,
            )

        if component_ids is not None and torch.unique(component_ids).numel() > 1:
            print("执行 Segmentation for heterogeneous dynamics：SAM/语义部件优先，运动特征辅助细分。")
        elif target_patches > 1:
            print("未检测到多组件点级语义标签，执行 motion-aware K-Means 异质动态伪分割。")
            component_ids = self._run_kmeans(motion_features, target_patches)
        else:
            zeros = torch.zeros((num_points,), dtype=torch.long, device=device)
            return zeros, torch.zeros((1,), dtype=torch.long, device=device)

        patch_budget = self._allocate_patch_budget_per_component(
            component_ids,
            target_patches,
        )
        patch_ids = torch.empty((num_points,), dtype=torch.long, device=device)
        patch_component_ids = []
        next_patch_id = 0
        for component_id in torch.unique(component_ids, sorted=True):
            component_id_int = int(component_id.item())
            component_mask = (component_ids == component_id)
            component_indices = torch.where(component_mask)[0]
            component_patch_count = min(
                patch_budget[component_id_int],
                int(component_indices.numel()),
            )
            subpatch_ids = self._run_kmeans(
                motion_features[component_mask],
                component_patch_count,
            )
            for subpatch_id in torch.unique(subpatch_ids, sorted=True):
                subpatch_mask = (subpatch_ids == subpatch_id)
                patch_ids[component_indices[subpatch_mask]] = next_patch_id
                patch_component_ids.append(component_id_int)
                next_patch_id += 1

        patch_component_ids = torch.tensor(
            patch_component_ids,
            dtype=torch.long,
            device=device,
        )
        print(f"异质动态分割完成：{torch.unique(component_ids).numel()} 个组件，{next_patch_id} 个 surface patches。")
        return patch_ids, patch_component_ids

    def _enforce_c0_continuity(self, patch_cps_dict, patch_adjacency, num_u_cp, num_v_cp):
        """
        通过平均化共享边界的控制点来强制实现 C0 连续性。
        Args:
            patch_cps_dict (dict): {patch_id: control_points_tensor}，其中每个值形状为 [1, U, V, T, 3]。
            patch_adjacency (dict): {(patch_i, patch_j): "U"/"V"}。
        Returns:
            dict: 更新后的控制点字典。
        """
        # >>> MODIFIED: 保留原接口，但不再依赖 create_from_ply_dict 中的硬编码 patch 编号语义。
        for (p_id1, p_id2), direction in patch_adjacency.items():
            if p_id1 not in patch_cps_dict or p_id2 not in patch_cps_dict:
                continue

            cp1 = patch_cps_dict[p_id1]
            cp2 = patch_cps_dict[p_id2]

            if direction == "U":
                boundary1 = cp1[:, -1, :, :, :]
                boundary2 = cp2[:, 0, :, :, :]
                avg_boundary = (boundary1 + boundary2) / 2.0
                cp1[:, -1, :, :, :] = avg_boundary
                cp2[:, 0, :, :, :] = avg_boundary

            elif direction == "V":
                boundary1 = cp1[:, :, -1, :, :]
                boundary2 = cp2[:, :, 0, :, :]
                avg_boundary = (boundary1 + boundary2) / 2.0
                cp1[:, :, -1, :, :] = avg_boundary
                cp2[:, :, 0, :, :] = avg_boundary

        return patch_cps_dict

    def _normalize_uv_to_unit_square(self, uv: torch.Tensor) -> torch.Tensor:
        """
        将任意局部 UV 重新归一化到 [0,1]^2。
        """
        # >>> MODIFIED: 多 patch 模式下，每个 patch 使用局部归一化后的 UV，
        # 避免直接裁剪全局 UV 导致采样域过窄、最小二乘拟合病态。
        if uv.numel() == 0:
            return uv
        minv, _ = uv.min(dim=0, keepdim=True)
        maxv, _ = uv.max(dim=0, keepdim=True)
        span = (maxv - minv).clamp_min(1e-6)
        return (uv - minv) / span

    def _build_patch_adjacency_from_global_uv(
        self,
        global_uv: torch.Tensor,
        patch_ids: torch.Tensor,
        patch_component_ids: torch.Tensor = None,
    ):
        """
        基于全局 UV 包围盒自动推断 patch 邻接关系。
        异质动态组件之间保持独立，不建立 C0 连续性约束。
        返回 {(pid_i, pid_j): "U"/"V"}。
        """
        # >>> MODIFIED: 用几何/参数域关系自动推断邻接，替代原先对 KMeans label 的硬编码。
        unique_patch_ids = torch.unique(patch_ids)
        if unique_patch_ids.numel() <= 1:
            return {}

        patch_boxes = {}
        widths = []
        heights = []
        for pid in unique_patch_ids.tolist():
            uv_patch = global_uv[patch_ids == pid]
            if uv_patch.numel() == 0:
                continue
            umin = uv_patch[:, 0].min().item()
            umax = uv_patch[:, 0].max().item()
            vmin = uv_patch[:, 1].min().item()
            vmax = uv_patch[:, 1].max().item()
            ucen = uv_patch[:, 0].mean().item()
            vcen = uv_patch[:, 1].mean().item()
            patch_boxes[pid] = dict(umin=umin, umax=umax, vmin=vmin, vmax=vmax, ucen=ucen, vcen=vcen)
            widths.append(max(umax - umin, 1e-6))
            heights.append(max(vmax - vmin, 1e-6))

        if len(patch_boxes) <= 1:
            return {}

        width_ref = float(np.median(widths)) if len(widths) > 0 else 0.25
        height_ref = float(np.median(heights)) if len(heights) > 0 else 0.25
        tol_u = max(0.15 * width_ref, 0.03)
        tol_v = max(0.15 * height_ref, 0.03)

        adjacency = {}
        box_items = list(patch_boxes.items())
        for i in range(len(box_items)):
            pid_i, box_i = box_items[i]
            for j in range(i + 1, len(box_items)):
                pid_j, box_j = box_items[j]
                if patch_component_ids is not None:
                    if (
                        pid_i >= patch_component_ids.shape[0]
                        or pid_j >= patch_component_ids.shape[0]
                        or int(patch_component_ids[pid_i].item()) != int(patch_component_ids[pid_j].item())
                    ):
                        continue

                overlap_u = max(0.0, min(box_i["umax"], box_j["umax"]) - max(box_i["umin"], box_j["umin"]))
                overlap_v = max(0.0, min(box_i["vmax"], box_j["vmax"]) - max(box_i["vmin"], box_j["vmin"]))
                union_u = max(box_i["umax"], box_j["umax"]) - min(box_i["umin"], box_j["umin"]) + 1e-8
                union_v = max(box_i["vmax"], box_j["vmax"]) - min(box_i["vmin"], box_j["vmin"]) + 1e-8
                overlap_u_ratio = overlap_u / union_u
                overlap_v_ratio = overlap_v / union_v

                if overlap_v_ratio > 0.20:
                    gap_ij = abs(box_j["umin"] - box_i["umax"])
                    gap_ji = abs(box_i["umin"] - box_j["umax"])
                    gap_u = min(gap_ij, gap_ji)
                    if gap_u <= tol_u:
                        if box_i["ucen"] <= box_j["ucen"]:
                            adjacency[(pid_i, pid_j)] = "U"
                        else:
                            adjacency[(pid_j, pid_i)] = "U"

                if overlap_u_ratio > 0.20:
                    gap_ij = abs(box_j["vmin"] - box_i["vmax"])
                    gap_ji = abs(box_i["vmin"] - box_j["vmax"])
                    gap_v = min(gap_ij, gap_ji)
                    if gap_v <= tol_v:
                        if box_i["vcen"] <= box_j["vcen"]:
                            adjacency[(pid_i, pid_j)] = "V"
                        else:
                            adjacency[(pid_j, pid_i)] = "V"

        return adjacency

    def _compute_bspline_basis_levels(self, values: torch.Tensor, knots: torch.Tensor, degree: int, num_control_points: int):
        """
        使用 Cox-de Boor 递推计算从 0 阶到 degree 阶的全部 B-spline basis。
        返回一个列表 levels，其中 levels[p] 的形状都是 [N, num_control_points]。
        """
        x = values.reshape(-1).clamp(0.0, 1.0)
        knots = knots.to(dtype=values.dtype, device=values.device)
        x_column = x.unsqueeze(1)
        basis = (
            (x_column >= knots[:num_control_points])
            & (x_column < knots[1:num_control_points + 1])
        ).to(dtype=values.dtype)
        endpoint = x >= 1.0 - 1e-8
        if endpoint.any():
            basis[endpoint] = 0.0
            basis[endpoint, num_control_points - 1] = 1.0

        levels = [basis]
        for p in range(1, degree + 1):
            prev = levels[-1]
            denom_left = knots[p:p + num_control_points] - knots[:num_control_points]
            denom_right = (
                knots[p + 1:p + num_control_points + 1]
                - knots[1:num_control_points + 1]
            )
            valid_left = denom_left.abs() > 1e-12
            valid_right = denom_right.abs() > 1e-12
            safe_left = torch.where(valid_left, denom_left, torch.ones_like(denom_left))
            safe_right = torch.where(valid_right, denom_right, torch.ones_like(denom_right))
            shifted_prev = torch.cat([prev[:, 1:], torch.zeros_like(prev[:, :1])], dim=1)
            left = ((x_column - knots[:num_control_points]) / safe_left) * prev
            right = (
                (knots[p + 1:p + num_control_points + 1] - x_column) / safe_right
            ) * shifted_prev
            levels.append(left * valid_left + right * valid_right)

        return levels

    def _compute_bspline_basis_from_knots(
        self,
        values: torch.Tensor,
        knots: torch.Tensor,
        degree: int,
        num_control_points: int,
    ):
        """Compute only basis values for the training hot path."""
        return self._compute_bspline_basis_levels(
            values, knots, degree, num_control_points
        )[degree]

    def _compute_bspline_basis_and_derivatives_from_knots(
        self,
        values: torch.Tensor,
        knots: torch.Tensor,
        degree: int,
        num_control_points: int,
    ):
        """
        基于 open-uniform knots 计算完整 B-spline basis 及其一、二阶导。
        返回:
            basis, d_basis, dd_basis: [N, num_control_points]
        """
        levels = self._compute_bspline_basis_levels(values, knots, degree, num_control_points)
        basis = levels[degree]

        if degree == 0:
            d_basis = torch.zeros_like(basis)
            dd_basis = torch.zeros_like(basis)
            return basis, d_basis, dd_basis

        def derivative_from_level(level, target_degree):
            denom_left = (
                knots[target_degree:target_degree + num_control_points]
                - knots[:num_control_points]
            )
            denom_right = (
                knots[target_degree + 1:target_degree + num_control_points + 1]
                - knots[1:num_control_points + 1]
            )
            valid_left = denom_left.abs() > 1e-12
            valid_right = denom_right.abs() > 1e-12
            safe_left = torch.where(valid_left, denom_left, torch.ones_like(denom_left))
            safe_right = torch.where(valid_right, denom_right, torch.ones_like(denom_right))
            shifted_level = torch.cat([level[:, 1:], torch.zeros_like(level[:, :1])], dim=1)
            return (
                target_degree * level / safe_left * valid_left
                - target_degree * shifted_level / safe_right * valid_right
            )

        prev = levels[degree - 1]
        d_basis = derivative_from_level(prev, degree)

        if degree == 1:
            dd_basis = torch.zeros_like(basis)
            return basis, d_basis, dd_basis

        d_prev = derivative_from_level(levels[degree - 2], degree - 1)
        dd_basis = derivative_from_level(d_prev, degree)
        return basis, d_basis, dd_basis
    # ============================ 新增辅助函数 结束 ============================



    def _orthonormalize_surface_frame(self, tangent_u: torch.Tensor, tangent_v: torch.Tensor, eps: float = 1e-8):
        """
        使用 Gram-Schmidt 构造局部曲面坐标系。
        返回旋转矩阵 R=[t_u, t_v, n]，三个基向量按列堆叠。
        """
        t_u = F.normalize(tangent_u, dim=-1, eps=eps)
        tangent_v_proj = tangent_v - torch.sum(tangent_v * t_u, dim=-1, keepdim=True) * t_u
        tv_norm = torch.norm(tangent_v_proj, dim=-1, keepdim=True)

        z_axis = torch.tensor([0.0, 0.0, 1.0], device=tangent_u.device, dtype=tangent_u.dtype).view(1, 3).expand_as(t_u)
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=tangent_u.device, dtype=tangent_u.dtype).view(1, 3).expand_as(t_u)
        use_y_axis = torch.abs(torch.sum(t_u * z_axis, dim=-1, keepdim=True)) > 0.95
        ref_axis = torch.where(use_y_axis, y_axis, z_axis)
        tangent_v_fallback = torch.cross(ref_axis, t_u, dim=-1)
        tangent_v_proj = torch.where(tv_norm > eps, tangent_v_proj, tangent_v_fallback)

        t_v = F.normalize(tangent_v_proj, dim=-1, eps=eps)
        n = F.normalize(torch.cross(t_u, t_v, dim=-1), dim=-1, eps=eps)
        t_v = F.normalize(torch.cross(n, t_u, dim=-1), dim=-1, eps=eps)

        return torch.stack([t_u, t_v, n], dim=-1)

    def _rotation_matrix_to_quaternion_wxyz(self, rot_mats: torch.Tensor, eps: float = 1e-8):
        """
        将旋转矩阵转换为四元数，输出格式为 [w, x, y, z]。
        rot_mats: [N, 3, 3]
        """
        rot_mats = rot_mats.to(dtype=torch.float32)
        n = rot_mats.shape[0]
        quats = torch.zeros((n, 4), device=rot_mats.device, dtype=rot_mats.dtype)

        m00 = rot_mats[:, 0, 0]
        m01 = rot_mats[:, 0, 1]
        m02 = rot_mats[:, 0, 2]
        m10 = rot_mats[:, 1, 0]
        m11 = rot_mats[:, 1, 1]
        m12 = rot_mats[:, 1, 2]
        m20 = rot_mats[:, 2, 0]
        m21 = rot_mats[:, 2, 1]
        m22 = rot_mats[:, 2, 2]

        trace = m00 + m11 + m22
        mask = trace > 0.0

        if mask.any():
            s = torch.sqrt(trace[mask] + 1.0 + eps) * 2.0
            quats[mask, 0] = 0.25 * s
            quats[mask, 1] = (m21[mask] - m12[mask]) / s
            quats[mask, 2] = (m02[mask] - m20[mask]) / s
            quats[mask, 3] = (m10[mask] - m01[mask]) / s

        mask_x = (~mask) & (m00 > m11) & (m00 > m22)
        if mask_x.any():
            s = torch.sqrt(1.0 + m00[mask_x] - m11[mask_x] - m22[mask_x] + eps) * 2.0
            quats[mask_x, 0] = (m21[mask_x] - m12[mask_x]) / s
            quats[mask_x, 1] = 0.25 * s
            quats[mask_x, 2] = (m01[mask_x] + m10[mask_x]) / s
            quats[mask_x, 3] = (m02[mask_x] + m20[mask_x]) / s

        mask_y = (~mask) & (~mask_x) & (m11 > m22)
        if mask_y.any():
            s = torch.sqrt(1.0 + m11[mask_y] - m00[mask_y] - m22[mask_y] + eps) * 2.0
            quats[mask_y, 0] = (m02[mask_y] - m20[mask_y]) / s
            quats[mask_y, 1] = (m01[mask_y] + m10[mask_y]) / s
            quats[mask_y, 2] = 0.25 * s
            quats[mask_y, 3] = (m12[mask_y] + m21[mask_y]) / s

        mask_z = (~mask) & (~mask_x) & (~mask_y)
        if mask_z.any():
            s = torch.sqrt(1.0 + m22[mask_z] - m00[mask_z] - m11[mask_z] + eps) * 2.0
            quats[mask_z, 0] = (m10[mask_z] - m01[mask_z]) / s
            quats[mask_z, 1] = (m02[mask_z] + m20[mask_z]) / s
            quats[mask_z, 2] = (m12[mask_z] + m21[mask_z]) / s
            quats[mask_z, 3] = 0.25 * s

        return F.normalize(quats, dim=-1, eps=eps).to(dtype=rot_mats.dtype)

    # ============================ 曲面几何初始化辅助函数 结束 ============================

    def _initialize_surface_aligned_rotation_and_scaling(
        self,
        gaussian_control_point: torch.Tensor,
        gaussian_uv: torch.Tensor,
        point_to_cp_idx: torch.Tensor,
        base_log_scales: torch.Tensor,
        gaussian_cp_group: torch.Tensor,
        t_ref: float = 0.0,
        chunk_size: int = 32768,
    ):
        """
        基于曲面局部微分几何初始化每个高斯的旋转与尺度。
        - 旋转：由局部切向量与法向量构造局部坐标系。
        - 尺度：在原始 KNN 初始化尺度的基础上，引入局部度量与曲率调制，
          使高斯在高曲率区域更贴合曲面，在法向方向更薄。

        对于静态背景或退化曲面，自动回退到原始各向同性初始化。
        """
        device = gaussian_uv.device
        dtype = gaussian_control_point.dtype
        num_points = gaussian_uv.shape[0]

        refined_scales = self.scaling_activation(base_log_scales).clone()
        rots = torch.zeros((num_points, 4), device=device, dtype=dtype)
        rots[:, 0] = 1.0

        if num_points == 0 or gaussian_control_point.numel() == 0:
            return self.scaling_inverse_activation(refined_scales.clamp_min(1e-12)), rots

        point_cp_group = gaussian_cp_group[point_to_cp_idx]
        dynamic_point_indices = torch.where(point_cp_group > 0)[0]
        if dynamic_point_indices.numel() == 0:
            return self.scaling_inverse_activation(refined_scales.clamp_min(1e-12)), rots

        t_tensor = torch.tensor([[t_ref]], device=device, dtype=dtype)
        B_t = self.BezierCoeff(t_tensor).to(dtype=dtype, device=device).squeeze(0)

        alpha_u = 0.5
        alpha_v = 0.5
        beta = 0.2
        eps = 1e-8

        for start in range(0, dynamic_point_indices.numel(), chunk_size):
            end = min(start + chunk_size, dynamic_point_indices.numel())
            point_idx_chunk = dynamic_point_indices[start:end]
            cp_idx_chunk = point_to_cp_idx[point_idx_chunk]

            cp_chunk = gaussian_control_point[cp_idx_chunk]
            uv_chunk = gaussian_uv[point_idx_chunk].to(dtype=dtype)
            base_scale_chunk = refined_scales[point_idx_chunk]

            cp_at_ref_t = torch.einsum('nuvtc,t->nuvc', cp_chunk, B_t)

            # >>> MODIFIED: 直接显式指定 U/V 两个方向的 knots 与控制点数，
            # 避免在 surface_u_order == surface_v_order 时发生歧义。
            B_u, dB_u, ddB_u = self._compute_bspline_basis_and_derivatives_from_knots(
                values=uv_chunk[:, :1],
                knots=self.knots_u.to(dtype=dtype, device=device),
                degree=self.surface_u_order,
                num_control_points=self.num_surface_u_control_points,
            )
            B_v, dB_v, ddB_v = self._compute_bspline_basis_and_derivatives_from_knots(
                values=uv_chunk[:, 1:2],
                knots=self.knots_v.to(dtype=dtype, device=device),
                degree=self.surface_v_order,
                num_control_points=self.num_surface_v_control_points,
            )

            tangent_u = torch.einsum('nuvc,nu,nv->nc', cp_at_ref_t, dB_u, B_v)
            tangent_v = torch.einsum('nuvc,nu,nv->nc', cp_at_ref_t, B_u, dB_v)
            ddu = torch.einsum('nuvc,nu,nv->nc', cp_at_ref_t, ddB_u, B_v)
            ddv = torch.einsum('nuvc,nu,nv->nc', cp_at_ref_t, B_u, ddB_v)

            cross_uv = torch.cross(tangent_u, tangent_v, dim=-1)
            valid_mask = (
                (torch.norm(tangent_u, dim=-1) > eps)
                & (torch.norm(tangent_v, dim=-1) > eps)
                & (torch.norm(cross_uv, dim=-1) > eps)
            )
            if not valid_mask.any():
                continue

            rot_mats = self._orthonormalize_surface_frame(tangent_u[valid_mask], tangent_v[valid_mask], eps=eps)
            rots[point_idx_chunk[valid_mask]] = self._rotation_matrix_to_quaternion_wxyz(rot_mats, eps=eps).to(dtype=dtype)

            normal_vec = rot_mats[:, :, 2]
            tangent_u_norm = torch.norm(tangent_u[valid_mask], dim=-1)
            tangent_v_norm = torch.norm(tangent_v[valid_mask], dim=-1)

            kappa_u = torch.abs(torch.sum(ddu[valid_mask] * normal_vec, dim=-1)) / (tangent_u_norm.square() + eps)
            kappa_v = torch.abs(torch.sum(ddv[valid_mask] * normal_vec, dim=-1)) / (tangent_v_norm.square() + eps)

            base_iso_scale = base_scale_chunk[valid_mask].mean(dim=-1)
            metric_ratio = torch.sqrt((tangent_u_norm + eps) / (tangent_v_norm + eps)).clamp(0.5, 2.0)

            scale_u = base_iso_scale * metric_ratio / (1.0 + alpha_u * kappa_u)
            scale_v = base_iso_scale / metric_ratio / (1.0 + alpha_v * kappa_v)
            scale_n = (0.5 * base_iso_scale) / (1.0 + beta * (kappa_u + kappa_v))

            min_scale = (0.25 * base_iso_scale).clamp_min(1e-6)
            max_scale = (2.50 * base_iso_scale).clamp_min(1e-6)
            scale_u = torch.clamp(scale_u, min=min_scale, max=max_scale)
            scale_v = torch.clamp(scale_v, min=min_scale, max=max_scale)
            scale_n = torch.clamp(scale_n, min=min_scale, max=max_scale)

            refined_scales[point_idx_chunk[valid_mask]] = torch.stack([scale_u, scale_v, scale_n], dim=-1)

        return self.scaling_inverse_activation(refined_scales.clamp_min(1e-12)), rots
    # ============================ 曲面几何初始化辅助函数 结束 ============================

    # for surface visualization
    @torch.no_grad()
    def _save_ply_xyzrgb(self, xyz, rgb, save_path):
        xyz = xyz.detach().cpu().float()
        rgb = rgb.detach().cpu().to(torch.uint8)

        with open(save_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {xyz.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(xyz, rgb):
                f.write(f"{p[0].item()} {p[1].item()} {p[2].item()} "
                        f"{int(c[0].item())} {int(c[1].item())} {int(c[2].item())}\n")

    # for surface visualization
    @torch.no_grad()
    def _sample_static_surface(self, ctrl_uv, res=50):
        """
        ctrl_uv: [U, V, 3]
        return: [res*res, 3]
        """
        device = ctrl_uv.device
        dtype = ctrl_uv.dtype

        u = torch.linspace(0, 1, res, device=device, dtype=dtype)
        v = torch.linspace(0, 1, res, device=device, dtype=dtype)
        uu, vv = torch.meshgrid(u, v, indexing="ij")

        Bu, Bv = self.SurfCoeff(uu.reshape(-1, 1), vv.reshape(-1, 1))  # [N,U], [N,V]
        xyz = torch.einsum("uvc,nu,nv->nc", ctrl_uv, Bu, Bv)
        return xyz

    # for surface visualization
    @torch.no_grad()
    def _sample_dynamic_surface(self, ctrl_uvt, t=0.0, res=50):
        """
        ctrl_uvt: [U, V, 4, 3]
        return: [res*res, 3]
        """
        device = ctrl_uvt.device
        dtype = ctrl_uvt.dtype

        Bt = self.BezierCoeff(
            torch.tensor([[t]], device=device, dtype=dtype)
        ).squeeze(0)  # [4]

        # 先在时间维度上求当前时刻的曲面控制点
        ctrl_uv = torch.einsum("uvtc,t->uvc", ctrl_uvt, Bt)  # [U,V,3]

        # 再在(u,v)上采样
        return self._sample_static_surface(ctrl_uv, res=res)

    # for surface visualization
    @torch.no_grad()
    def export_init_surfaces(self, save_dir, object_id=None, t_list=(0.0, 0.5, 1.0), res=50):
        """
        从 self._control_points 导出初始化曲面
        只建议看动态物体: object_id > 0
        """
        os.makedirs(save_dir, exist_ok=True)

        cps = self._control_points.detach()          # [M,U,V,4,3]
        ids = self._cp_idx_to_ids.detach()           # [M,2] = [object_id, surface_id/patch_id]
        groups = self._cp_group.detach()             # [M]

        # 只看动态曲面
        mask = (groups > 0)
        if object_id is not None:
            mask = mask & (ids[:, 0] == object_id)

        selected_idx = torch.where(mask)[0]

        for idx in selected_idx.tolist():
            oid = int(ids[idx, 0].item())
            sid = int(ids[idx, 1].item())
            ctrl = cps[idx]  # [U,V,4,3]

            for t in t_list:
                surf_xyz = self._sample_dynamic_surface(ctrl, t=t, res=res)

                # 控制点也一起导出，便于看 cage
                ctrl_xyz = ctrl.reshape(-1, 3)

                surf_rgb = torch.full(
                    (surf_xyz.shape[0], 3), 180, device=surf_xyz.device, dtype=torch.uint8
                )
                ctrl_rgb = torch.tensor(
                    [[255, 0, 0]], device=ctrl_xyz.device, dtype=torch.uint8
                ).repeat(ctrl_xyz.shape[0], 1)

                xyz = torch.cat([surf_xyz, ctrl_xyz], dim=0)
                rgb = torch.cat([surf_rgb, ctrl_rgb], dim=0)

                save_path = os.path.join(save_dir, f"obj{oid}_patch{sid}_t{t:.2f}.ply")
                self._save_ply_xyzrgb(xyz, rgb, save_path)

                print(f"[exported] {save_path}")

    def create_from_ply_dict(self, ply_dict, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale
        gaussian_control_point = []
        gaussian_color = []
        gaussian_scales = []
        gaussian_group = []

        gaussian_uv = []
        gaussian_cp_group = []
        cp_idx_to_ids = []
        point_to_cp_idx = []
        cp_idx_counter = 0

        obj_num = len(ply_dict.keys()) - 1
        # >>> MODIFIED: 不再把时间控制点数写死为 4，统一使用 self.num_control_points。
        self.trajectory_cp_tensor = torch.zeros([obj_num, self.num_control_points, 3], dtype=torch.float32).cuda()
        self.cp_recorded_timestamp_2_bezier_t = torch.zeros([obj_num, self.num_control_points, 1], dtype=torch.float32).cuda()

        for k, v in ply_dict.items():
            if k == 'bkgd':
                bg_xyz = torch.from_numpy(v['xyz_array']).float().cuda()
                bg_color = torch.from_numpy(v['colors_array']).float().cuda()
                control_points = bg_xyz

                # >>> MODIFIED: 背景时间维不再写死 4，统一使用 self.num_control_points。
                control_points = control_points.unsqueeze(1).unsqueeze(1).unsqueeze(1)
                control_points = control_points.repeat(
                    1,
                    self.num_surface_u_control_points,
                    self.num_surface_v_control_points,
                    self.num_control_points,
                    1,
                )

                r_max = 100000
                r_min = 2
                num_sph = self.random_init_point

                theta = 2 * torch.pi * torch.rand(num_sph)
                phi = (torch.pi / 2 * 0.99 * torch.rand(num_sph)) ** 1.5
                s = torch.rand(num_sph)
                r_1 = s * 1 / r_min + (1 - s) * 1 / r_max
                r = 1 / r_1
                pts_sph = torch.stack(
                    [r * torch.cos(theta) * torch.cos(phi), r * torch.sin(theta) * torch.cos(phi), r * torch.sin(phi)],
                    dim=-1,
                ).cuda()
                pts_sph = pts_sph.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(
                    1,
                    self.num_surface_u_control_points,
                    self.num_surface_v_control_points,
                    self.num_control_points,
                    1,
                )
                pts_sph[:, :, :, :, 2] = -pts_sph[:, :, :, :, 2] + 1

                control_points = torch.cat([control_points, pts_sph], dim=0)
                bg_color = torch.cat([bg_color, torch.zeros(pts_sph.shape[0], 3).float().cuda()], dim=0)

                uv_parameterized = torch.zeros([control_points.shape[0], 2], dtype=torch.float32).cuda()

                gaussian_control_point.append(control_points)
                gaussian_uv.append(uv_parameterized)

                dist2 = torch.clamp_min(distCUDA2(control_points[:, 0, 0, 0, :]), 0.0000001)
                scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)
                gaussian_scales.append(scales)
                gaussian_color.append(bg_color)
                gaussian_group.append(torch.zeros((control_points.shape[0]), dtype=torch.int, device="cuda"))
                gaussian_cp_group.append(torch.zeros((control_points.shape[0]), dtype=torch.int, device="cuda"))

                surface_ids = torch.arange(uv_parameterized.shape[0], dtype=torch.long, device="cuda")
                object_ids = torch.full((uv_parameterized.shape[0],), self.group_num, dtype=torch.long, device="cuda")
                ids_map = torch.stack([object_ids, surface_ids], dim=1)
                cp_idx_to_ids.append(ids_map)

                point_map = torch.arange(
                    cp_idx_counter,
                    cp_idx_counter + uv_parameterized.shape[0],
                    dtype=torch.long,
                    device="cuda",
                )
                point_to_cp_idx.append(point_map)
                cp_idx_counter += uv_parameterized.shape[0]
            else:
                if v['xyz_offset'] is None or v['xyz_offset'].shape[0] < 30:
                    continue

                if self.dynamic_mode == "Bezier":
                    self.group_num += 1
                    xyz_offset = torch.from_numpy(v['xyz_offset']).float().cuda()
                    obj_color = torch.from_numpy(v['colors_array']).float().cuda()
                    trajectory = torch.from_numpy(v['trajectory']).float().cuda()

                    t_parameterized = self.get_chord_len_parametrization(trajectory)
                    trajectory_cp = self.generate_control_points(
                        trajectory.unsqueeze(0),
                        t_parameterized.unsqueeze(1)
                    ).squeeze(0)

                    for round in range(100):
                        t_samples = torch.linspace(0, 1, 10000, dtype=trajectory.dtype, device=trajectory.device)
                        Bernstein_mat = self.BezierCoeff(t_samples.unsqueeze(1))
                        curve_samples = Bernstein_mat @ trajectory_cp

                        diff = trajectory.unsqueeze(1) - curve_samples.unsqueeze(0)
                        diff_sq = torch.sum(diff ** 2, dim=-1)
                        indices = torch.argmin(diff_sq, dim=-1)
                        bezier_t = t_samples[indices]

                        new_cp = self.generate_control_points(
                            trajectory.unsqueeze(0),
                            bezier_t.unsqueeze(1)
                        ).squeeze(0)

                        diff_ctrl = new_cp - trajectory_cp
                        trajectory_cp = new_cp
                        if torch.norm(diff_ctrl, dim=-1).max() < 1e-6:
                            print("round: ", round)
                            break
                        if round == 99:
                            print("round99: ", round)

                    self.trajectory_dict[self.group_num] = dict(
                        zip(v['timestamp_list'], [torch.tensor(pos, device='cuda') for pos in trajectory.tolist()])
                    )
                    self.trajectory_cp_tensor[self.group_num - 1] = trajectory_cp
                    self.world_bezier_time_map[self.group_num] = dict(
                        zip(v['timestamp_list'], [torch.tensor([[t]], device='cuda') for t in bezier_t.tolist()])
                    )
                    self.cp_recorded_timestamp_2_bezier_t[self.group_num - 1] = torch.linalg.lstsq(
                        self.BezierCoeff(torch.tensor(v['timestamp_list']).unsqueeze(1).cuda()),
                        bezier_t.unsqueeze(1).cuda()
                    ).solution
                    self.exist_range[self.group_num] = [v['timestamp_list'][0], v['timestamp_list'][-1]]

                    mean_shape_xyz = xyz_offset.mean(dim=1)

                    # Segmentation for heterogeneous dynamics:
                    # 优先消费预处理阶段由 SAM 等模型生成并投影到 3D 点的部件标签。
                    # 没有标签时，num_patches > 1 仍可使用运动感知伪分割。
                    patch_ids_for_points, patch_component_ids = self._segmentation_for_heterogeneous_dynamics(
                        xyz_offset=xyz_offset,
                        mean_shape_xyz=mean_shape_xyz,
                        obj_record=v,
                        num_patches=self.num_patches,
                    )

                    if torch.unique(patch_ids_for_points).numel() > 1:
                        print(f"为物体 {self.group_num} 拟合 {torch.unique(patch_ids_for_points).numel()} 个动态曲面...")
                        unique_patch_ids = torch.unique(patch_ids_for_points)

                        # >>> MODIFIED: 先得到全局一致的 UV 方向，再在每个 patch 内做局部归一化。
                        global_uv_parameterized = self.pca_uv_init(mean_shape_xyz)
                        local_uv_per_point = torch.zeros_like(global_uv_parameterized)

                        patch_cps_dict = {}
                        for i, patch_id in enumerate(unique_patch_ids):
                            print(f"为物体 {self.group_num} 的面片 {patch_id.item()} ({i+1}/{unique_patch_ids.numel()}) 拟合控制点...")
                            point_mask = (patch_ids_for_points == patch_id)

                            xyz_offset_patch = xyz_offset[point_mask]
                            global_uv_patch = global_uv_parameterized[point_mask]
                            local_uv_patch = self._normalize_uv_to_unit_square(global_uv_patch)
                            local_uv_per_point[point_mask] = local_uv_patch

                            u_parameterized_patch = local_uv_patch[:, :1]
                            v_parameterized_patch = local_uv_patch[:, 1:2]

                            if xyz_offset_patch.shape[0] == 0:
                                continue

                            # >>> MODIFIED: 即便 patch 点数较少，也不再直接丢弃，
                            # 而是允许通过 pinv 做欠定最小二乘拟合，避免点被过滤掉。
                            if xyz_offset_patch.shape[0] < self.num_surface_u_control_points * self.num_surface_v_control_points:
                                print(f"警告：面片 {patch_id.item()} 点数较少，将使用 pinv 欠定拟合。")

                            surface_cps_per_frame_patch = []
                            for frame_idx in range(xyz_offset_patch.shape[1]):
                                cp_per_frame = self.generate_surface_control_points(
                                    xyz_offset_patch[:, frame_idx, :],
                                    u_parameterized_patch,
                                    v_parameterized_patch,
                                    None,
                                )
                                surface_cps_per_frame_patch.append(cp_per_frame)

                            surface_cps_patch = torch.stack(surface_cps_per_frame_patch, dim=0)

                            surface_cp_cps_patch = self.generate_control_points(
                                surface_cps_patch.view(surface_cps_patch.shape[0], -1, 3).permute(1, 0, 2),
                                bezier_t.unsqueeze(1)
                            ).view(
                                self.num_surface_u_control_points,
                                self.num_surface_v_control_points,
                                self.num_control_points,
                                3,
                            ).unsqueeze(0)

                            patch_cps_dict[patch_id.item()] = surface_cp_cps_patch

                        # >>> MODIFIED: 用全局 UV 包围盒自动推断 patch 邻接，替代硬编码 adjacency。
                        adjacency = self._build_patch_adjacency_from_global_uv(
                            global_uv_parameterized,
                            patch_ids_for_points,
                            patch_component_ids=patch_component_ids,
                        )
                        if len(adjacency) > 0:
                            print("基于全局 UV 自动推断 patch 邻接，并执行 C0 连续性约束...")
                            patch_cps_dict = self._enforce_c0_continuity(
                                patch_cps_dict,
                                adjacency,
                                self.num_surface_u_control_points,
                                self.num_surface_v_control_points,
                            )

                        valid_patch_ids = [pid.item() for pid in unique_patch_ids if pid.item() in patch_cps_dict]
                        if len(valid_patch_ids) == 0:
                            print(f"警告：物体 {self.group_num} 的多 patch 拟合失败，回退到单曲面初始化。")
                            uv_parameterized = global_uv_parameterized
                            u_parameterized = uv_parameterized[:, :1]
                            v_parameterized = uv_parameterized[:, 1:2]

                            surface_cps_per_frame = []
                            for frame_idx in range(xyz_offset.shape[1]):
                                cp_per_frame = self.generate_surface_control_points(
                                    xyz_offset[:, frame_idx, :],
                                    u_parameterized,
                                    v_parameterized,
                                    None,
                                )
                                surface_cps_per_frame.append(cp_per_frame)

                            surface_cps = torch.stack(surface_cps_per_frame, dim=0)
                            surface_cp_cps = self.generate_control_points(
                                surface_cps.view(surface_cps.shape[0], -1, 3).permute(1, 0, 2),
                                bezier_t.unsqueeze(1)
                            ).view(
                                self.num_surface_u_control_points,
                                self.num_surface_v_control_points,
                                self.num_control_points,
                                3,
                            ).unsqueeze(0)

                            gaussian_control_point.append(surface_cp_cps)
                            gaussian_uv.append(uv_parameterized)
                            gaussian_color.append(obj_color)
                            gaussian_group.append(torch.full((xyz_offset.shape[0],), self.group_num, dtype=torch.int, device="cuda"))
                            gaussian_cp_group.append(torch.full((1,), self.group_num, dtype=torch.int, device="cuda"))

                            dist2 = torch.clamp_min(distCUDA2(mean_shape_xyz), 0.0000001)
                            scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)
                            gaussian_scales.append(scales)

                            surface_ids = torch.zeros(1, dtype=torch.long, device="cuda")
                            object_ids = torch.full((1,), self.group_num, dtype=torch.long, device="cuda")
                            ids_map = torch.stack([object_ids, surface_ids], dim=1)
                            cp_idx_to_ids.append(ids_map)

                            point_map = torch.full((xyz_offset.shape[0],), cp_idx_counter, dtype=torch.long, device="cuda")
                            point_to_cp_idx.append(point_map)
                            cp_idx_counter += 1
                        else:
                            valid_patch_ids_tensor = torch.tensor(valid_patch_ids, dtype=torch.long, device="cuda")

                            for pid in unique_patch_ids:
                                if pid.item() in patch_cps_dict:
                                    gaussian_control_point.append(patch_cps_dict[pid.item()])

                            num_patches_found = len(valid_patch_ids)
                            gaussian_cp_group.append(torch.full((num_patches_found,), self.group_num, dtype=torch.int, device="cuda"))

                            patch_object_ids = torch.full((num_patches_found,), self.group_num, dtype=torch.long, device='cuda')
                            ids_map = torch.stack([patch_object_ids, valid_patch_ids_tensor], dim=1)
                            cp_idx_to_ids.append(ids_map)

                            point_map_for_object = torch.full((global_uv_parameterized.shape[0],), -1, dtype=torch.long, device="cuda")
                            patch_id_to_new_master_idx = {pid: cp_idx_counter + i for i, pid in enumerate(valid_patch_ids)}
                            for pid_val, mid_val in patch_id_to_new_master_idx.items():
                                mask = (patch_ids_for_points == pid_val)
                                point_map_for_object[mask] = mid_val

                            unassigned_mask = (point_map_for_object < 0)
                            if unassigned_mask.any():
                                fallback_pid = valid_patch_ids[0]
                                point_map_for_object[unassigned_mask] = patch_id_to_new_master_idx[fallback_pid]
                                local_uv_per_point[unassigned_mask] = 0.5
                                print(f"警告：物体 {self.group_num} 存在未分配点，已回退映射到 patch {fallback_pid}。")

                            point_to_cp_idx.append(point_map_for_object)
                            cp_idx_counter += num_patches_found

                            gaussian_color.append(obj_color)
                            gaussian_uv.append(local_uv_per_point)
                            gaussian_group.append(torch.full((xyz_offset.shape[0],), self.group_num, dtype=torch.int, device="cuda"))

                            dist2 = torch.clamp_min(distCUDA2(mean_shape_xyz), 0.0000001)
                            scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)
                            gaussian_scales.append(scales)

                    else:
                        print(f"为物体 {self.group_num} 拟合单个动态曲面...")
                        uv_parameterized = self.pca_uv_init(mean_shape_xyz)
                        u_parameterized = uv_parameterized[:, :1]
                        v_parameterized = uv_parameterized[:, 1:2]

                        surface_cps_per_frame = []
                        for frame_idx in range(xyz_offset.shape[1]):
                            cp_per_frame = self.generate_surface_control_points(
                                xyz_offset[:, frame_idx, :],
                                u_parameterized,
                                v_parameterized,
                                None,
                            )
                            surface_cps_per_frame.append(cp_per_frame)

                        surface_cps = torch.stack(surface_cps_per_frame, dim=0)
                        surface_cp_cps = self.generate_control_points(
                            surface_cps.view(surface_cps.shape[0], -1, 3).permute(1, 0, 2),
                            bezier_t.unsqueeze(1)
                        ).view(
                            self.num_surface_u_control_points,
                            self.num_surface_v_control_points,
                            self.num_control_points,
                            3,
                        ).unsqueeze(0)

                        gaussian_control_point.append(surface_cp_cps)
                        gaussian_uv.append(uv_parameterized)
                        gaussian_color.append(obj_color)
                        gaussian_group.append(torch.full((xyz_offset.shape[0],), self.group_num, dtype=torch.int, device="cuda"))
                        gaussian_cp_group.append(torch.full((1,), self.group_num, dtype=torch.int, device="cuda"))

                        dist2 = torch.clamp_min(distCUDA2(mean_shape_xyz), 0.0000001)
                        scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)
                        gaussian_scales.append(scales)

                        surface_ids = torch.zeros(1, dtype=torch.long, device="cuda")
                        object_ids = torch.full((1,), self.group_num, dtype=torch.long, device="cuda")
                        ids_map = torch.stack([object_ids, surface_ids], dim=1)
                        cp_idx_to_ids.append(ids_map)

                        point_map = torch.full((xyz_offset.shape[0],), cp_idx_counter, dtype=torch.long, device="cuda")
                        point_to_cp_idx.append(point_map)
                        cp_idx_counter += 1

        gaussian_control_point = torch.cat(gaussian_control_point, dim=0)
        gaussian_color = torch.cat(gaussian_color, dim=0)
        gaussian_scales = torch.cat(gaussian_scales, dim=0)
        gaussian_group = torch.cat(gaussian_group, dim=0).cuda()
        gaussian_cp_group = torch.cat(gaussian_cp_group, dim=0).cuda()

        if self.use_low_rank:
            print(f"\n执行塔克分解 (Tucker Decomposition) 至秩 R={self.low_rank_decom_order}...")
            decomposed_cores, decomposed_factors = self.batch_tucker_decomposition(
                gaussian_control_point,
                rank=self.low_rank_decom_order,
                batch_size=1024,
            )
        else:
            print("\n不执行塔克分解，直接使用原始控制点参数化。")
            decomposed_cores, decomposed_factors = None, None

        gaussian_uv = torch.cat(gaussian_uv, dim=0)
        cp_idx_to_ids = torch.cat(cp_idx_to_ids, dim=0)
        point_to_cp_idx = torch.cat(point_to_cp_idx, dim=0).cuda()

        print("\n正在为动态物体优化创建索引重映射表...")
        dynamic_cp_mask = (gaussian_cp_group > 0)
        original_dynamic_cp_indices = torch.where(dynamic_cp_mask)[0]

        num_total_cps = gaussian_control_point.shape[0]
        cp_remap_table = torch.full((num_total_cps,), -1, dtype=torch.long, device="cuda")
        cp_remap_table[original_dynamic_cp_indices] = torch.arange(
            len(original_dynamic_cp_indices), device="cuda"
        )
        print(f"优化完成。共找到 {len(original_dynamic_cp_indices)} 组动态控制点。")

        fused_color = RGB2SH(gaussian_color)
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        logging.info("Number of points at initialization: {}".format(gaussian_control_point.shape[0]))

        gaussian_scales, rots = self._initialize_surface_aligned_rotation_and_scaling(
            gaussian_control_point=gaussian_control_point,
            gaussian_uv=gaussian_uv,
            point_to_cp_idx=point_to_cp_idx,
            base_log_scales=gaussian_scales,
            gaussian_cp_group=gaussian_cp_group,
            t_ref=0.0,
        )

        opacities = inverse_sigmoid(self.opa_init * torch.ones((gaussian_uv.shape[0], 1), dtype=torch.float, device="cuda"))

        control_points_requires_grad = (not self.use_low_rank) or self.optimize_static_control_points
        self._control_points = nn.Parameter(
            gaussian_control_point.requires_grad_(control_points_requires_grad)
        )  # [M, U, V, T, 3]
        if self.use_low_rank:
            self._core_tensors = nn.Parameter(decomposed_cores.requires_grad_(True))
            self._factor_U = nn.Parameter(
                decomposed_factors[0].requires_grad_(self.optimize_low_rank_factors)
            )
            self._factor_V = nn.Parameter(
                decomposed_factors[1].requires_grad_(self.optimize_low_rank_factors)
            )
            self._factor_T = nn.Parameter(
                decomposed_factors[2].requires_grad_(self.optimize_low_rank_factors)
            )
        else:
            empty = torch.empty(0, device=gaussian_control_point.device, dtype=gaussian_control_point.dtype)
            self._core_tensors = empty
            self._factor_U = empty
            self._factor_V = empty
            self._factor_T = empty
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))  # [N, 1, 3]
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))  # [N, 15, 3]
        self._scaling = nn.Parameter(gaussian_scales.requires_grad_(True))  # [N, 3]
        self._rotation = nn.Parameter(rots.requires_grad_(True))  # [N, 4]
        self._opacity = nn.Parameter(opacities.requires_grad_(True))  # [N, 1]
        self._group = gaussian_group.requires_grad_(False)  # [N]
        self.max_radii2D = torch.zeros((gaussian_uv.shape[0]), device="cuda")  # [N]

        # ------------------------------------------
        # addition for Bezier surface
        self._gaussian_uv = nn.Parameter(gaussian_uv.requires_grad_(self.optimize_surface_uv))  # [N, 2]
        self._cp_idx_to_ids = cp_idx_to_ids.requires_grad_(False)  # [M, 2],[object_id, surface_id/patch_id]
        self._cp_group = gaussian_cp_group.requires_grad_(False)  # [M]
        self._point_to_cp_idx = point_to_cp_idx.requires_grad_(False)  # [N]
        # ------------------------------------------

        # --- [优化] 将掩码和重映射表保存为模型属性 (缓冲区) ---
        # 使用 register_buffer，这样它们会随模型移动(如 .to(device))，但不会被视为可训练参数。
        self._dynamic_cp_mask = dynamic_cp_mask.requires_grad_(False)  # [M]
        self._cp_remap_table = cp_remap_table.requires_grad_(False)  # [N]
        # ------------------------------------------

        self.trajectory_cp_tensor = nn.Parameter(self.trajectory_cp_tensor, requires_grad=True)  # [num_object, num_control_points, 3]，不同 object 中心轨迹对应的控制点
        self.cp_recorded_timestamp_2_bezier_t = nn.Parameter(self.cp_recorded_timestamp_2_bezier_t, requires_grad=True)  # [num_object, num_control_points, 1]

        self._rebuild_group_point_indices()

    def batch_tucker_decomposition(self, tensor, rank, batch_size=1024, device='cuda'):
        """
        以节省内存的方式，分批次对大张量执行塔克分解，为所有样本找到共享的因子矩阵。
        """
        if tl is None:
            raise ImportError("低秩模式需要 tensorly，请先执行: pip install tensorly")
        print(f"\n开始执行分批次塔克分解 (Batch Size: {batch_size})...")
        m_dim, u_dim, v_dim, t_dim, _ = tensor.shape
        
        if isinstance(rank, int):
            rank = [rank, rank, rank]
        
        # 1. 分批次计算协方差矩阵
        cov_u = torch.zeros((u_dim, u_dim), device=device)
        cov_v = torch.zeros((v_dim, v_dim), device=device)
        cov_t = torch.zeros((t_dim, t_dim), device=device)
        
        num_batches = (m_dim + batch_size - 1) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, m_dim)
            batch = tensor[start_idx:end_idx]
            
            for k in range(3): # 遍历 x, y, z
                batch_k = batch[..., k]
                cov_u += tl.unfold(batch_k, 1) @ tl.unfold(batch_k, 1).T
                cov_v += tl.unfold(batch_k, 2) @ tl.unfold(batch_k, 2).T
                cov_t += tl.unfold(batch_k, 3) @ tl.unfold(batch_k, 3).T
        
        # 2. 特征分解找到因子矩阵
        _, u_vecs = torch.linalg.eigh(cov_u)
        _, v_vecs = torch.linalg.eigh(cov_v)
        _, t_vecs = torch.linalg.eigh(cov_t)

        factor_u = u_vecs[:, -rank[0]:]
        factor_v = v_vecs[:, -rank[1]:]
        factor_t = t_vecs[:, -rank[2]:]
        factors = [factor_u, factor_v, factor_t]
        print("  - 共享因子矩阵计算完成。")

        # 3. 分批次投影获得核心张量
        core_tensors = torch.zeros([m_dim, rank[0], rank[1], rank[2], 3], device=device)
        factors_t = [tl.transpose(f) for f in factors]
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, m_dim)
            batch = tensor[start_idx:end_idx]
            core_tensors[start_idx:end_idx] = tl.tenalg.multi_mode_dot(batch, factors_t, modes=[1, 2, 3])
        print("  - 核心张量计算完成。")
        
        return core_tensors, factors

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_gaussian_uv.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_gaussian_uv.shape[0], 1), device="cuda")

        self.optimize_surface_uv = getattr(
            training_args, "optimize_surface_uv", self.optimize_surface_uv
        )
        self.optimize_low_rank_factors = getattr(
            training_args, "optimize_low_rank_factors", self.optimize_low_rank_factors
        )
        self.optimize_static_control_points = getattr(
            training_args, "optimize_static_control_points", self.optimize_static_control_points
        )
        self.surface_uv_lr_scale = getattr(training_args, "surface_uv_lr_scale", 0.02)

        self._gaussian_uv.requires_grad_(self.optimize_surface_uv)
        self._control_points.requires_grad_(
            (not self.use_low_rank) or self.optimize_static_control_points
        )
        if self.use_low_rank:
            self._core_tensors.requires_grad_(True)
            self._factor_U.requires_grad_(self.optimize_low_rank_factors)
            self._factor_V.requires_grad_(self.optimize_low_rank_factors)
            self._factor_T.requires_grad_(self.optimize_low_rank_factors)

        l = [
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]
        if (not self.use_low_rank) or self.optimize_static_control_points:
            l.append({
                'params': [self._control_points],
                'lr': training_args.position_lr_init * self.spatial_lr_scale,
                "name": "control_points",
            })
        if self.use_low_rank:
            l.append({
                'params': [self._core_tensors],
                'lr': training_args.position_lr_init * self.spatial_lr_scale,
                "name": "core_tensors",
            })
            if self.optimize_low_rank_factors:
                factor_lr = training_args.position_lr_init * self.spatial_lr_scale * 0.1
                l.extend([
                    {'params': [self._factor_U], 'lr': factor_lr, "name": "factor_U"},
                    {'params': [self._factor_V], 'lr': factor_lr, "name": "factor_V"},
                    {'params': [self._factor_T], 'lr': factor_lr, "name": "factor_T"},
                ])
        if self.optimize_surface_uv:
            l.append({
                'params': [self._gaussian_uv],
                'lr': training_args.position_lr_init * self.spatial_lr_scale * self.surface_uv_lr_scale,
                "name": "gaussian_uv",
            })
        self._dynamic_surface_basis_cache = None

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.cp_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.iterations)

        bezier_l = [
            {'params': self.trajectory_cp_tensor, 'lr': training_args.traj_pos_lr_init * self.spatial_lr_scale, "name": "traj"},
            {'params': self.cp_recorded_timestamp_2_bezier_t, 'lr': training_args.t_lr, "name": "bezier_param"}
        ]
        self.bezier_optimizer = torch.optim.Adam(bezier_l, lr=0.0, eps=1e-15)
        self.traj_cp_scheduler_args = get_expon_lr_func(lr_init=training_args.traj_pos_lr_init * self.spatial_lr_scale,
                                                        lr_final=training_args.traj_pos_lr_final * self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.iterations)

        if self.dynamic_mode == "DeformableGS":
            self.deform.train_setting(training_args)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "control_points":
                param_group['lr'] = self.cp_scheduler_args(iteration)
            elif param_group["name"] == "gaussian_uv":
                param_group['lr'] = self.surface_uv_lr_scale * self.cp_scheduler_args(iteration)
            elif param_group["name"] == "core_tensors":
                param_group['lr'] = self.cp_scheduler_args(iteration)
            elif param_group["name"] in ["factor_U", "factor_V", "factor_T"]:
                param_group['lr'] = 0.1 * self.cp_scheduler_args(iteration)
        for param_group in self.bezier_optimizer.param_groups:
            if param_group["name"] == "traj":
                lr = self.traj_cp_scheduler_args(iteration)
                param_group['lr'] = lr
        if self.dynamic_mode == "DeformableGS":
            self.deform.update_learning_rate(iteration)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * self.opa_init))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, param_to_mask_map: dict):
        """
        根据提供的【掩码字典】，剪枝优化器中的每个参数及其状态。
        这个新版本可以为不同尺寸的参数（如 master_control_points 和 scaling）应用不同的掩码。

        Args:
            param_to_mask_map (dict): 字典，键是参数名 (e.g., "master_control_points"),
                                      值是对应的布尔掩码 (True代表保留)。
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            param_name = group["name"]
            
            # 检查这个参数是否在我们的剪枝计划中
            if param_name in param_to_mask_map:
                mask = param_to_mask_map[param_name] # 获取专属掩码
                
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    # 使用专属掩码对 Adam 的状态 (momentum, variance) 进行切片
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    # 关键步骤：先删除旧参数的引用，再创建并关联新参数
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[param_name] = group["params"][0]
                else:
                    # 如果没有优化器状态，直接剪枝参数
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[param_name] = group["params"][0]
            else:
                # 如果此参数不在剪枝计划中 (例如 trajectory_cp_tensor)，则保持原样
                optimizable_tensors[param_name] = group["params"][0]

        return optimizable_tensors

    def prune_points(self, mask):
        """
        根据掩码剪枝点。此函数已为“曲面/面片”数据结构完全重构。
        执行逻辑：1. 剪枝点级数据 -> 2. 识别并剪枝无用的曲面 -> 3. 重映射主索引 -> 4. 更新优化器
        """
        valid_points_mask = ~mask

        # --- 步骤 1: 剪枝所有“点”级别 (Point-Level) 的张量 ---
        self._point_to_cp_idx = self._point_to_cp_idx[valid_points_mask]
        self._group = self._group[valid_points_mask]
        # 同样剪枝其他辅助张量
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        # --- 步骤 2: "垃圾回收" - 识别并剪枝无用的“曲面”级别张量 ---
        # a. 找出剪枝后，仍然被点引用的所有主索引
        used_cp_indices = torch.unique(self._point_to_cp_idx)

        # b. 创建一个用于剪枝“曲面”级别张量的掩码 (长度为 M)
        num_master_surfaces_before = self._control_points.shape[0]
        valid_master_mask = torch.zeros(num_master_surfaces_before, dtype=torch.bool, device="cuda")
        valid_master_mask[used_cp_indices] = True

        # c. 剪枝“曲面”级别的非可学习张量
        self._cp_idx_to_ids = self._cp_idx_to_ids[valid_master_mask]
        self._cp_group = self._cp_group[valid_master_mask]

        # --- 步骤 3: 关键！重映射主索引 (Remap Master Indices) ---
        # 因为我们从 _master_control_points 中删除了一些行，所以剩下行的索引会发生改变。
        # 我们需要更新 _point_to_master_idx 中的所有值，让它们指向正确的新位置。
        remap_table = torch.full((num_master_surfaces_before,), -1, dtype=torch.long, device="cuda")
        remap_table[valid_master_mask] = torch.arange(valid_master_mask.sum(), device="cuda", dtype=torch.long)
        self._point_to_cp_idx = remap_table[self._point_to_cp_idx]

        # --- 步骤 4: 更新优化器中的所有可学习参数 ---
        # a. 创建一个字典，为不同级别的参数指定不同的剪枝掩码
        param_to_mask_map = {
            # -- 点级别参数 (使用 valid_points_mask) --
            "f_dc": valid_points_mask,
            "f_rest": valid_points_mask,
            "opacity": valid_points_mask,
            "scaling": valid_points_mask,
            "rotation": valid_points_mask,
            "gaussian_uv": valid_points_mask,
            # -- 曲面级别参数 (使用 valid_master_mask) --
            "control_points": valid_master_mask,
            "core_tensors": valid_master_mask
        }

        optimizable_tensors = self._prune_optimizer(param_to_mask_map)

        if "control_points" in optimizable_tensors:
            self._control_points = optimizable_tensors["control_points"]
        else:
            self._control_points = nn.Parameter(
                self._control_points[valid_master_mask].requires_grad_(
                    self._control_points.requires_grad
                )
            )
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if "gaussian_uv" in optimizable_tensors:
            self._gaussian_uv = optimizable_tensors["gaussian_uv"]
        else:
            self._gaussian_uv = nn.Parameter(
                self._gaussian_uv[valid_points_mask].requires_grad_(
                    self._gaussian_uv.requires_grad
                )
            )
        if self.use_low_rank:
            if "core_tensors" in optimizable_tensors:
                self._core_tensors = optimizable_tensors["core_tensors"]
            else:
                self._core_tensors = nn.Parameter(
                    self._core_tensors[valid_master_mask].requires_grad_(
                        self._core_tensors.requires_grad
                    )
                )

        # ===============================================================
        #
        #               >>> 新增修改部分 <<<
        #
        # --- [关键同步步骤] ---
        # 由于控制点已被剪枝和重排，必须重建用于快速索引的掩码和重映射表。
        print("剪枝后，正在重建动态物体索引优化表...")
        self._update_dynamic_indexing_helpers()
        #
        # ===============================================================

        self._rebuild_group_point_indices()

    def _update_dynamic_indexing_helpers(self):
        """
        根据当前的模型状态 (_cp_group, _control_points) 重新计算
        _dynamic_cp_mask 和 cp_remap_table。
        这个函数应该在任何修改了控制点数量或顺序的操作后被调用。
        """
        # 1. 创建掩码
        self._dynamic_cp_mask = (self._cp_group > 0)

        # 2. 找到动态控制点的当前索引
        original_dynamic_cp_indices = torch.where(self._dynamic_cp_mask)[0]

        # 3. 创建并填充重映射表
        num_total_cps = self._control_points.shape[0]
        self._cp_remap_table = torch.full((num_total_cps,), -1, dtype=torch.long, device="cuda")
        self._cp_remap_table[original_dynamic_cp_indices] = torch.arange(
            len(original_dynamic_cp_indices), device="cuda"
        )

    def cat_tensors_to_optimizer(self, tensors_dict: dict):
        """
        将新创建的张量（来自密集化操作）拼接到优化器中现有的参数上，并正确扩展优化器状态。
        【已修改】：此版本可以处理只扩展部分参数组的情况。
        """
        # optimizable_tensors 用于收集所有参数（无论是否被修改）的最终状态
        optimizable_tensors = {}
        
        # 遍历优化器中的每一个参数组
        for group in self.optimizer.param_groups:
            # 断言检查，确保每个组只有一个参数张量，这符合3DGS的常规设置
            assert len(group["params"]) == 1
            
            # --- 关键修改：安全地获取待扩展张量 ---
            # 使用 .get() 方法代替直接索引 [ ]。
            # 如果在 tensors_dict 中找不到当前参数组的名字，get()会返回 None，而不是抛出错误。
            extension_tensor = tensors_dict.get(group["name"])

            # 只有当确实有需要扩展的数据时，才执行拼接操作
            if extension_tensor is not None and extension_tensor.shape[0] > 0:
                # 获取当前参数的优化器状态 (例如 Adam 的 momentum 和 variance)
                stored_state = self.optimizer.state.get(group['params'][0], None)
                
                if stored_state is not None:
                    # 将优化器状态也进行扩展，新添加部分的状态初始化为0
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    # 关键的“参数替换”步骤，以确保优化器能正确跟踪新张量
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                else:
                    # 如果该参数还没有优化器状态，直接拼接参数本身
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                
                # 将更新后的参数保存到返回字典中
                optimizable_tensors[group["name"]] = group["params"][0]
            
            else:
                # --- 新增逻辑 ---
                # 如果 tensors_dict 中没有为这个参数组提供扩展数据 (例如 clone 时对 control_points)
                # 那么我们什么都不做，直接将原始参数保留下来。
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, 
                              new_features_dc, 
                              new_features_rest, 
                              new_opacities, 
                              new_scaling,
                              new_rotation, 
                              new_gaussian_uv,
                              new_control_points=None,
                              # >>> 新增修改 <<<
                              # 增加 new_core_tensors 参数，用于接收新的核心张量
                              new_core_tensors=None
                              ):
        d = {
            "control_points": new_control_points,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "gaussian_uv": new_gaussian_uv,  # addition for Bezier surface
            # >>> 新增修改 <<<
            # 将 new_core_tensors 添加到待处理的字典中，修复之前的 "No" bug
            "core_tensors": new_core_tensors
            }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        if "control_points" in optimizable_tensors:
            self._control_points = optimizable_tensors["control_points"]
        elif new_control_points is not None and new_control_points.numel() > 0:
            self._control_points = nn.Parameter(
                torch.cat((self._control_points, new_control_points), dim=0).requires_grad_(
                    self._control_points.requires_grad
                )
            )
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if "gaussian_uv" in optimizable_tensors:
            self._gaussian_uv = optimizable_tensors["gaussian_uv"]
        elif new_gaussian_uv is not None and new_gaussian_uv.numel() > 0:
            self._gaussian_uv = nn.Parameter(
                torch.cat((self._gaussian_uv, new_gaussian_uv), dim=0).requires_grad_(
                    self._gaussian_uv.requires_grad
                )
            )
        if self.use_low_rank:
            if "core_tensors" in optimizable_tensors:
                self._core_tensors = optimizable_tensors["core_tensors"]
            elif new_core_tensors is not None and new_core_tensors.numel() > 0:
                self._core_tensors = nn.Parameter(
                    torch.cat((self._core_tensors, new_core_tensors), dim=0).requires_grad_(
                        self._core_tensors.requires_grad
                    )
                )

        # ------------------------------------------
        # change for Bezier surface
        # self.xyz_gradient_accum = torch.zeros((self.get_control_points.shape[0], 1), device="cuda")
        # self.denom = torch.zeros((self.get_control_points.shape[0], 1), device="cuda")
        # self.max_radii2D = torch.zeros((self.get_control_points.shape[0]), device="cuda")

        self.xyz_gradient_accum = torch.zeros((self.get_gaussian_uv.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_gaussian_uv.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_gaussian_uv.shape[0]), device="cuda")
        # ------------------------------------------

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        [混合策略重构 - 已修复元数据唯一性错误]
        - 对静态点: 克隆点和它的控制点（创建一个新的独立曲面，并分配新ID）。
        - 对动态点: 只克隆点（UV坐标和属性），让它依附于同一个旧的曲面。
        """
        all_points_control_points = self._control_points[self._point_to_cp_idx]
        if self.contract:
            scale_factor = all_points_control_points[:, 0, 0, 0, :].norm(dim=-1) * scene_extent - 1
            scale_factor = torch.where(scale_factor <= 1, 1, scale_factor) / scene_extent
        else:
            scale_factor = torch.ones_like(all_points_control_points[:, 0, 0, 0, :]) / scene_extent

        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent * scale_factor)

        # 初始化用于存储新数据的空张量
        new_control_points_list = []
        # >>> 新增修改 <<<
        # 为新的核心张量创建列表
        new_core_tensors_list = []
        new_cp_idx_to_ids_list = []
        new_cp_group_list = []
        new_point_level_tensors = {
            "f_dc": [], "f_rest": [], "opacity": [], "scaling": [],
            "rotation": [], "gaussian_uv": [], "group": [], "point_to_cp_idx": []
        }

        # --- 1. 处理静态点的克隆 ---
        static_clone_mask = torch.logical_and(selected_pts_mask, self._group == 0)
        if static_clone_mask.any():
            # 为每个被克隆的静态点创建一套新的、独立的控制点
            new_static_cps = self._control_points[self._point_to_cp_idx[static_clone_mask]]
            new_control_points_list.append(new_static_cps)

            if self.use_low_rank:
                new_static_cores = self._core_tensors[self._point_to_cp_idx[static_clone_mask]]
                new_core_tensors_list.append(new_static_cores)

            # 新控制点的主索引将从现有控制点总数之后开始
            num_cps_before = self._control_points.shape[0]
            num_new_static_points = static_clone_mask.sum()
            new_static_point_to_cp_idx = torch.arange(num_cps_before, num_cps_before + num_new_static_points, device="cuda")
            new_point_level_tensors["point_to_cp_idx"].append(new_static_point_to_cp_idx)

            # [关键修复] 为新创建的静态曲面生成全新的、唯一的 surface_id
            # 统计当前背景曲面的总数，作为新ID的起始值
            num_bkgd_surfaces = (self._cp_group == 0).sum()
            
            # 新的 object_id 仍然是 0 (背景组)
            new_object_ids = torch.zeros(num_new_static_points, dtype=torch.long, device="cuda")
            
            # 新的 surface_id 从现有总数开始递增
            new_surface_ids = torch.arange(num_bkgd_surfaces, num_bkgd_surfaces + num_new_static_points, dtype=torch.long, device="cuda")
            
            # 组合成新的 _cp_idx_to_ids 元数据
            new_static_cp_ids = torch.stack([new_object_ids, new_surface_ids], dim=1)
            new_cp_idx_to_ids_list.append(new_static_cp_ids)

            # 新曲面的组别仍然是 0
            new_cp_group = torch.zeros(num_new_static_points, dtype=torch.long, device="cuda")
            new_cp_group_list.append(new_cp_group)

            # 复制点级别的其他属性
            new_point_level_tensors["f_dc"].append(self._features_dc[static_clone_mask])
            new_point_level_tensors["f_rest"].append(self._features_rest[static_clone_mask])
            new_point_level_tensors["opacity"].append(self._opacity[static_clone_mask])
            new_point_level_tensors["scaling"].append(self._scaling[static_clone_mask])
            new_point_level_tensors["rotation"].append(self._rotation[static_clone_mask])
            new_point_level_tensors["gaussian_uv"].append(self._gaussian_uv[static_clone_mask])
            new_point_level_tensors["group"].append(self._group[static_clone_mask])

        # --- 2. 处理动态点的克隆 (此部分逻辑正确，无需修改) ---
        dynamic_clone_mask = torch.logical_and(selected_pts_mask, self._group > 0)
        if dynamic_clone_mask.any():
            # 动态点不创建新的控制点，而是复用父点的主索引
            new_dynamic_point_to_cp_idx = self._point_to_cp_idx[dynamic_clone_mask]
            new_point_level_tensors["point_to_cp_idx"].append(new_dynamic_point_to_cp_idx)

            # 复制点级别的其他属性
            new_point_level_tensors["f_dc"].append(self._features_dc[dynamic_clone_mask])
            new_point_level_tensors["f_rest"].append(self._features_rest[dynamic_clone_mask])
            new_point_level_tensors["opacity"].append(self._opacity[dynamic_clone_mask])
            new_point_level_tensors["scaling"].append(self._scaling[dynamic_clone_mask])
            new_point_level_tensors["rotation"].append(self._rotation[dynamic_clone_mask])
            new_point_level_tensors["gaussian_uv"].append(self._gaussian_uv[dynamic_clone_mask])
            new_point_level_tensors["group"].append(self._group[dynamic_clone_mask])

        # --- 3. 整合并更新数据 ---
        if new_control_points_list or any(t for t in new_point_level_tensors.values() if t):
            new_control_points = torch.cat(new_control_points_list, dim=0) if new_control_points_list else torch.empty(0, device="cuda")
            # >>> 新增修改 <<<
            # 整合新的核心张量
            new_core_tensors = torch.cat(new_core_tensors_list, dim=0) if new_core_tensors_list else None

            # 整合所有点级别的数据
            self.densification_postfix(
                new_features_dc=torch.cat(new_point_level_tensors["f_dc"], dim=0),
                new_features_rest=torch.cat(new_point_level_tensors["f_rest"], dim=0),
                new_opacities=torch.cat(new_point_level_tensors["opacity"], dim=0),
                new_scaling=torch.cat(new_point_level_tensors["scaling"], dim=0),
                new_rotation=torch.cat(new_point_level_tensors["rotation"], dim=0),
                new_gaussian_uv=torch.cat(new_point_level_tensors["gaussian_uv"], dim=0),
                new_control_points=new_control_points,
                # >>> 新增修改 <<<
                # 将新的核心张量传递下去
                new_core_tensors=new_core_tensors
            )
            self._group = torch.cat((self._group, torch.cat(new_point_level_tensors["group"], dim=0)), dim=0)
            self._point_to_cp_idx = torch.cat((self._point_to_cp_idx, torch.cat(new_point_level_tensors["point_to_cp_idx"], dim=0)), dim=0)
            
            if new_cp_idx_to_ids_list:
                self._cp_idx_to_ids = torch.cat([self._cp_idx_to_ids, torch.cat(new_cp_idx_to_ids_list, dim=0)], dim=0)
            if new_cp_group_list:
                self._cp_group = torch.cat([self._cp_group, torch.cat(new_cp_group_list, dim=0)], dim=0)

            # ===============================================================
            #
            #               >>> 新增修改部分 <<<
            #
            # --- [关键同步步骤] ---
            # 因为添加了新的控制点，必须重建索引优化表。
            print("克隆后，正在重建动态物体索引优化表...")
            self._update_dynamic_indexing_helpers()
            #
            # ===============================================================

            self._rebuild_group_point_indices()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        [混合策略重构 - 已修复元数据唯一性错误]
        - 对静态点: 分裂点和它的控制点（在世界空间中创建N个新的独立曲面，并分配新ID）。
        - 对动态点: 只分裂UV坐标，让N个新点依附于同一个旧的曲面，并缩小尺寸。
        """
        n_init_points = self._gaussian_uv.shape[0]
        all_points_control_points = self._control_points[self._point_to_cp_idx]
        
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        if self.contract:
            scale_factor = all_points_control_points[:, 0, 0, 0, :].norm(dim=-1) * scene_extent - 1
            scale_factor = torch.where(scale_factor <= 1, 1, scale_factor) / scene_extent
        else:
            scale_factor = torch.ones_like(all_points_control_points[:, 0, 0, 0, :]) / scene_extent

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent * scale_factor)
        
        decay_factor = N * 0.8

        # 初始化用于存储新数据的空张量
        new_control_points_list = []
        # >>> 新增修改 <<<
        new_core_tensors_list = []
        new_cp_idx_to_ids_list = []
        new_cp_group_list = []
        new_point_level_tensors = {
            "f_dc": [], "f_rest": [], "opacity": [], "scaling": [],
            "rotation": [], "gaussian_uv": [], "group": [], "point_to_cp_idx": []
        }

        # --- 1. 处理静态点的分裂 ---
        static_split_mask = torch.logical_and(selected_pts_mask, self._group == 0)
        if static_split_mask.any():
            # 采用类似原始3DGS的分裂方式：在世界空间中采样
            stds = self.get_scaling[static_split_mask].repeat_interleave(N, dim=0)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[static_split_mask]).repeat_interleave(N, dim=0)

            # 父辈的世界坐标
            parent_xyz = all_points_control_points[static_split_mask][:, 0, 0, 0, :]
            
            # 新的世界坐标
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + parent_xyz.repeat_interleave(N, dim=0)

            # 为每个分裂出的新点创建一套新的、独立的（退化）控制点
            new_static_cps = new_xyz.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, self.num_surface_u_control_points, self.num_surface_v_control_points, self.num_control_points, 1)
            new_control_points_list.append(new_static_cps)

            if self.use_low_rank:
                num_new_static_cps = new_static_cps.shape[0]
                rank = self._core_tensors.shape[1]
                new_static_cores = torch.zeros(
                    (num_new_static_cps, rank, rank, rank, 3),
                    device="cuda",
                    dtype=self._core_tensors.dtype,
                )
                new_core_tensors_list.append(new_static_cores)

            # [关键修复] 为N个分裂出的新曲面生成全新的、唯一的 surface_id
            num_new_static_points_total = static_split_mask.sum() * N
            
            # 统计当前背景曲面的总数，作为新ID的起始值
            # 注意：需要考虑在同一个批次中，clone操作可能已经向列表中添加了新的控制点
            num_cps_added_by_clone = sum(cps.shape[0] for cps in new_control_points_list) - new_static_cps.shape[0]
            num_bkgd_surfaces_before_split = (self._cp_group == 0).sum() + num_cps_added_by_clone
            
            # 新的 object_id 仍然是 0
            new_object_ids = torch.zeros(num_new_static_points_total, dtype=torch.long, device="cuda")
            # 新的 surface_id 从现有总数开始递增
            new_surface_ids = torch.arange(num_bkgd_surfaces_before_split, num_bkgd_surfaces_before_split + num_new_static_points_total, dtype=torch.long, device="cuda")
            
            new_static_cp_ids = torch.stack([new_object_ids, new_surface_ids], dim=1)
            new_cp_idx_to_ids_list.append(new_static_cp_ids)

            # 新曲面的组别仍然是 0
            new_static_cp_group = torch.zeros(num_new_static_points_total, dtype=torch.long, device="cuda")
            new_cp_group_list.append(new_static_cp_group)

            # 新控制点的主索引
            num_cps_before = self._control_points.shape[0] + num_cps_added_by_clone
            new_static_point_to_cp_idx = torch.arange(num_cps_before, num_cps_before + num_new_static_points_total, device="cuda")
            new_point_level_tensors["point_to_cp_idx"].append(new_static_point_to_cp_idx)

            # 复制/更新其他点级属性（全部统一为 repeat_interleave）
            new_point_level_tensors["scaling"].append(self.scaling_inverse_activation(self.get_scaling[static_split_mask].repeat_interleave(N, dim=0) / decay_factor))
            new_point_level_tensors["rotation"].append(self._rotation[static_split_mask].repeat_interleave(N, dim=0))
            new_point_level_tensors["f_dc"].append(self._features_dc[static_split_mask].repeat_interleave(N, dim=0))
            new_point_level_tensors["f_rest"].append(self._features_rest[static_split_mask].repeat_interleave(N, dim=0))
            new_point_level_tensors["opacity"].append(self._opacity[static_split_mask].repeat_interleave(N, dim=0))
            new_point_level_tensors["group"].append(self._group[static_split_mask].repeat_interleave(N))
            new_point_level_tensors["gaussian_uv"].append(torch.zeros((num_new_static_points_total, 2), device="cuda"))

        # --- 2. 处理动态点的分裂 (此部分逻辑正确，无需修改) ---
        dynamic_split_mask = torch.logical_and(selected_pts_mask, self._group > 0)
        if dynamic_split_mask.any():
            # 动态点分裂UV坐标，而不是世界坐标
            parent_uvs = self._gaussian_uv[dynamic_split_mask]
            # 在父辈UV坐标附近添加少量噪声来创建新的UV坐标
            uv_noise = (torch.rand_like(parent_uvs.repeat_interleave(N, dim=0)) - 0.5) * 0.1 # 可调超参数：噪声范围
            new_dynamic_uvs = parent_uvs.repeat_interleave(N, dim=0) + uv_noise
            new_dynamic_uvs.clamp_(0.0, 1.0) # 确保UV在[0,1]范围内
            new_point_level_tensors["gaussian_uv"].append(new_dynamic_uvs)

            # 新点复用父辈的主索引
            new_dynamic_point_to_cp_idx = self._point_to_cp_idx[dynamic_split_mask].repeat_interleave(N)
            new_point_level_tensors["point_to_cp_idx"].append(new_dynamic_point_to_cp_idx)

            # 复制/更新其他点级属性（全部统一为 repeat_interleave）
            new_point_level_tensors["scaling"].append(self.scaling_inverse_activation(self.get_scaling[dynamic_split_mask].repeat_interleave(N, dim=0) / decay_factor))
            new_point_level_tensors["rotation"].append(self._rotation[dynamic_split_mask].repeat_interleave(N, dim=0))
            new_point_level_tensors["f_dc"].append(self._features_dc[dynamic_split_mask].repeat_interleave(N, dim=0))
            new_point_level_tensors["f_rest"].append(self._features_rest[dynamic_split_mask].repeat_interleave(N, dim=0))
            new_point_level_tensors["opacity"].append(self._opacity[dynamic_split_mask].repeat_interleave(N, dim=0))
            new_point_level_tensors["group"].append(self._group[dynamic_split_mask].repeat_interleave(N))

        # --- 3. 整合、更新并剪枝父辈 ---
        if new_control_points_list or any(tensors for tensors in new_point_level_tensors.values() if tensors):
            new_control_points = torch.cat(new_control_points_list, dim=0) if new_control_points_list else torch.empty(0, device="cuda")
            # >>> 新增修改 <<<
            # 整合新的核心张量
            new_core_tensors = torch.cat(new_core_tensors_list, dim=0) if new_core_tensors_list else None
            
            # 整合所有新创建的点级数据
            self.densification_postfix(
                new_features_dc=torch.cat(new_point_level_tensors["f_dc"], dim=0),
                new_features_rest=torch.cat(new_point_level_tensors["f_rest"], dim=0),
                new_opacities=torch.cat(new_point_level_tensors["opacity"], dim=0),
                new_scaling=torch.cat(new_point_level_tensors["scaling"], dim=0),
                new_rotation=torch.cat(new_point_level_tensors["rotation"], dim=0),
                new_gaussian_uv=torch.cat(new_point_level_tensors["gaussian_uv"], dim=0),
                new_control_points=new_control_points,
                # >>> 新增修改 <<<
                # 将新的核心张量传递下去
                new_core_tensors=new_core_tensors
            )
            self._group = torch.cat((self._group, torch.cat(new_point_level_tensors["group"], dim=0)), dim=0)
            self._point_to_cp_idx = torch.cat((self._point_to_cp_idx, torch.cat(new_point_level_tensors["point_to_cp_idx"], dim=0)), dim=0)
            
            if new_cp_idx_to_ids_list:
                self._cp_idx_to_ids = torch.cat([self._cp_idx_to_ids, torch.cat(new_cp_idx_to_ids_list, dim=0)], dim=0)
            if new_cp_group_list:
                self._cp_group = torch.cat([self._cp_group, torch.cat(new_cp_group_list, dim=0)], dim=0)
            
            # 创建一个掩码来剪掉被分裂的父辈点
            num_new_points = sum(t.shape[0] for t in new_point_level_tensors["gaussian_uv"])
            prune_filter = torch.cat((selected_pts_mask, torch.zeros(num_new_points, device="cuda", dtype=bool)))
            self.prune_points(prune_filter)

    def densify_and_prune(self, velocity, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        all_points_control_points = self._control_points[self._point_to_cp_idx]
        if self.contract:
            # ------------------------------------------
            # change for Bezier surface
            # scale_factor = self._control_points[:, 0, :].norm(dim=-1)*scene_extent-1

            scale_factor = all_points_control_points[:, 0, 0, 0, :].norm(dim=-1)*extent-1
            # ------------------------------------------
            scale_factor = torch.where(scale_factor<=1, 1, scale_factor)/extent
        else:
            # scale_factor = torch.ones_like(self._control_points)[:, 0, :]/scene_extent

            scale_factor = torch.ones_like(all_points_control_points)[:, 0, 0, 0, :]/extent

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > self.big_point_threshold * extent * scale_factor  ## ori 0.1
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1




    # 固定的系数公式，不可学习, eq(4)
    def get_bezier_coefficient(self):
        # ts: (T, 1) -> M: (T, num_control_points)
        n = self.num_control_points - 1  # 公式(4)中的n，4个控制点对应n=3阶
        ks = torch.arange(self.num_control_points, dtype=torch.float32, device="cuda")  # 公式(4)中的i，从0到3

        def bezier_coeff(ts: torch.Tensor):
            t_pow_k = ts ** ks # (T, num_cp)  # 公式(4)中的t^i
            one_minus_t_pow_n_minus_k = (1.0 - ts) ** (n - ks)  # 公式(4)中的(1-t)^(n-i)
            M = self.binomial_coefs * t_pow_k * one_minus_t_pow_n_minus_k  # (T, num_cp)  # 公式(4)中的C(n,i) * t^i * (1-t)^(n-i)
            return M

        return bezier_coeff

    # ------------------------------------------
    # addition for Bezier surface
    # ------------------------------------------


    def get_bspline_surface_coefficient(self):
        """
        构造一个函数：给定 u, v ∈ [0,1]，返回 B-spline 曲面在两个方向上的完整基函数矩阵。
        返回:
            B_u: [N, num_surface_u_control_points]
            B_v: [N, num_surface_v_control_points]
        """
        # >>> MODIFIED: 不再使用只支持 order+1 维局部基函数的矩阵版本，
        # 改为基于 open-uniform knots 的完整 Cox-de Boor basis。
        def bspline_surf_coeffs(u: torch.Tensor, v: torch.Tensor):
            B_u = self._compute_bspline_basis_from_knots(
                values=u,
                knots=self.knots_u.to(dtype=u.dtype, device=u.device),
                degree=self.surface_u_order,
                num_control_points=self.num_surface_u_control_points,
            )
            B_v = self._compute_bspline_basis_from_knots(
                values=v,
                knots=self.knots_v.to(dtype=v.dtype, device=v.device),
                degree=self.surface_v_order,
                num_control_points=self.num_surface_v_control_points,
            )
            return B_u, B_v

        return bspline_surf_coeffs

    # bezier位置求导，即速度公式
    def get_d_bezier_coefficient(self):
        # ts: (T, 1) -> dM: (T, 1, num_control_points)
        n = self.num_control_points - 1
        lower_ks = torch.arange(n, dtype=torch.float32, device="cuda")
        lower_coefs = torch.tensor(
            [math.comb(n - 1, k) for k in range(n)],
            dtype=torch.float32,
            device="cuda",
        )

        def d_bezier_coeff(ts: torch.Tensor):
            if n == 0:
                return ts.new_zeros((ts.shape[0], 1, 1))
            lower_basis = (
                lower_coefs
                * ts ** lower_ks
                * (1.0 - ts) ** (n - 1 - lower_ks)
            )
            zeros = ts.new_zeros((ts.shape[0], 1))
            derivative = n * (
                torch.cat([zeros, lower_basis], dim=1)
                - torch.cat([lower_basis, zeros], dim=1)
            )
            return derivative.unsqueeze(1)

        return d_bezier_coeff

    # eq(21)，根据曲线位置反向求解控制点位置
    def generate_control_points(self, xyz_array, time_list):
        # xyz_array: [N, num_frames, 3]
        # time_list: [num_frames, 1]
        # Return: control_points: [N, num_control_points, 3]
        M = self.BezierCoeff(time_list).to(dtype=xyz_array.dtype, device=xyz_array.device).unsqueeze(0)  # [1, num_frames, num_control_points]
        MT = M.transpose(1, 2)  # [1, num_control_points, num_frames]
        MTM = torch.matmul(MT, M)  # B^T B, [1, num_control_points, num_control_points]
        MTM_inv = torch.linalg.pinv(MTM)  # (B^T B)^(−1), [1, num_control_points, num_control_points]
        control_points = torch.matmul(MTM_inv, torch.matmul(MT, xyz_array))  # P = (B^T B)^(−1) B^T X, [N, num_control_points, 3]
        return control_points

    # ------------------------------------------
    # addition for Bezier surface
    def generate_surface_control_points(self, xyz, u, v, time_list=None):
        """
        根据点云 xyz 及其参数化坐标 (u, v)，拟合单帧静态 B-spline 曲面的控制点。
        输入：
            xyz: [N, 3]
            u:   [N, 1]
            v:   [N, 1]
            time_list: 为兼容旧接口而保留，当前函数中不使用
        输出：
            surface_cp: [num_u, num_v, 3]
        """
        # >>> MODIFIED: time_list 保留仅为兼容 create_from_ply_dict 旧调用形式。
        # >>> MODIFIED: B_u/B_v 现在与真实控制点数量一致，不再隐式假设 num_cp == order + 1。
        B_u, B_v = self.SurfCoeff(u, v)
        B_u = B_u.to(dtype=xyz.dtype, device=xyz.device)
        B_v = B_v.to(dtype=xyz.dtype, device=xyz.device)

        B_uv = B_u.unsqueeze(2) * B_v.unsqueeze(1)  # [N, U, V]
        B = B_uv.view(B_uv.shape[0], -1)            # [N, U*V]

        BT = B.transpose(0, 1)
        BTB = torch.matmul(BT, B)
        BTB_inv = torch.linalg.pinv(BTB)
        surface_cp_flat = torch.matmul(BTB_inv, torch.matmul(BT, xyz))
        surface_cp = surface_cp_flat.view(
            self.num_surface_u_control_points,
            self.num_surface_v_control_points,
            3,
        )
        return surface_cp
    # ------------------------------------------

    def get_chord_len_parametrization(self, trajectory):
        # trajectory: [num_frames, 3]
        num_frames, _ = trajectory.shape
        if num_frames == 1:
            return torch.zeros((1), dtype=trajectory.dtype, device=trajectory.device)

        dist = torch.norm(trajectory[1:] - trajectory[:-1], dim=1) # [num_frames - 1], eq(19)
        s_0 = torch.zeros(1, dtype=trajectory.dtype, device=trajectory.device)
        s = torch.cat([s_0, torch.cumsum(dist, dim=0)]) # [num_frames]

        total_len = s[-1]
        eps = 1e-12

        if total_len < eps:
            t = torch.linspace(0, 1, num_frames, dtype=trajectory.dtype, device=trajectory.device)
        else:
            t = s / (total_len + eps)  # eq(20)

        return t


    # ------------------------------------------
    # addition for Bezier surface
    def pca_uv_init(self, points: torch.Tensor) -> torch.Tensor:
        """
        points: [N, 3]
        return: uv: [N, 2] in [0,1]^2
        """
        # >>> MODIFIED: 对极小样本/退化点集做更稳健的处理，避免 SVD 或归一化产生 NaN。
        if points.shape[0] == 0:
            return torch.empty((0, 2), dtype=points.dtype, device=points.device)
        if points.shape[0] == 1:
            return torch.zeros((1, 2), dtype=points.dtype, device=points.device)

        X = points - points.mean(dim=0, keepdim=True)
        try:
            _, _, Vh = torch.linalg.svd(X, full_matrices=False)
            V = Vh.transpose(0, 1)
            proj2 = X @ V[:, :2]
        except RuntimeError:
            proj2 = X[:, :2]

        minv, _ = proj2.min(dim=0, keepdim=True)
        maxv, _ = proj2.max(dim=0, keepdim=True)
        span = (maxv - minv).clamp_min(1e-8)
        uv = (proj2 - minv) / span
        return uv
    # ------------------------------------------
