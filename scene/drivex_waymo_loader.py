# Description: Load the EmerWaymo dataset for training and testing
# adapted from the PVG datareader for the data from EmerNeRF

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from scene.scene_utils import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly
from utils.graphics_utils import BasicPointCloud, focal2fov
from utils.general_utils import save_ply, load_ply
from utils.waymo_utils import get_obj_pose_tracking, build_pointcloud, build_bbox_mask
from utils.general_utils_drivex import quaternion_to_matrix_numpy
from collections import defaultdict
import cv2

image_filename_to_cam = lambda x: int(x.split('.')[0][-1])
image_filename_to_frame = lambda x: int(x.split('.')[0][:6])
DEFAULT_SAM_PART_SEGMENTATION_DIR = "seg_hetero_dynamics_npy"
SEGMENTATION_PAYLOAD_KEYS = (
    "labels",
    "sam_part_labels",
    "part_labels",
    "component_labels",
    "semantic_labels",
    "segment_ids",
    "segmentation",
    "masks",
)


def _use_seg_hetero_dynamics(args):
    value = getattr(args, "use_seg_hetero_dynamics", False)
    if isinstance(value, str):
        return value.lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _resolve_sam_part_segmentation_dir(args):
    segmentation_dir = getattr(args, "sam_part_segmentation_dir", None)
    if segmentation_dir is None:
        segmentation_dir = getattr(args, "heterogeneous_segmentation_dir", None)
    if segmentation_dir is None:
        segmentation_dir = DEFAULT_SAM_PART_SEGMENTATION_DIR
    if not os.path.isabs(segmentation_dir):
        segmentation_dir = os.path.join(args.source_path, segmentation_dir)
    return segmentation_dir


def _extract_segmentation_payload(loaded, path):
    if isinstance(loaded, np.lib.npyio.NpzFile):
        selected_key = next(
            (key for key in SEGMENTATION_PAYLOAD_KEYS if key in loaded.files),
            None,
        )
        if selected_key is None and len(loaded.files) == 1:
            selected_key = loaded.files[0]
        if selected_key is None:
            loaded.close()
            print(f"Warning: cannot determine the segmentation field in {path}.")
            return None
        payload = loaded[selected_key]
        loaded.close()
        return payload

    if isinstance(loaded, np.ndarray) and loaded.shape == () and loaded.dtype.kind == "O":
        payload = loaded.item()
        if isinstance(payload, dict):
            selected_key = next(
                (key for key in SEGMENTATION_PAYLOAD_KEYS if key in payload),
                None,
            )
            if selected_key is None:
                print(f"Warning: cannot determine the segmentation field in {path}.")
                return None
            return payload[selected_key]
    return loaded


def _load_numpy_segmentation(path, cache=None):
    if cache is not None and path in cache:
        return cache[path]
    try:
        try:
            loaded = np.load(path, allow_pickle=True, mmap_mode="r")
        except ValueError:
            # Object arrays cannot be memory-mapped.
            loaded = np.load(path, allow_pickle=True)
        payload = _extract_segmentation_payload(loaded, path)
    except (OSError, ValueError) as exc:
        print(f"Warning: cannot load SAM part segmentation {path}: {exc}")
        payload = None
    if cache is not None:
        cache[path] = payload
    return payload


def _coerce_view_segmentation_labels(payload, path):
    if payload is None:
        return None
    labels = np.asarray(payload)
    if labels.ndim == 2:
        return labels
    if labels.ndim != 3:
        print(f"Warning: expected a 2D label map or 3D mask stack in {path}, got {labels.shape}.")
        return None

    if labels.shape[0] < labels.shape[1] and labels.shape[0] < labels.shape[2]:
        scores = labels
        max_scores = scores.max(axis=0)
        return np.where(max_scores > 0, scores.argmax(axis=0) + 1, 0)
    if labels.shape[2] < labels.shape[0] and labels.shape[2] < labels.shape[1]:
        scores = labels
        max_scores = scores.max(axis=2)
        return np.where(max_scores > 0, scores.argmax(axis=2) + 1, 0)

    print(f"Warning: cannot infer the mask axis in {path}, got {labels.shape}.")
    return None


def _load_direct_object_part_labels(segmentation_dir, object_id, num_points, cache):
    point_label_dir = os.path.join(segmentation_dir, "point_labels")
    stems = [f"obj_{object_id:03d}", f"points3D_obj_{object_id:03d}"]
    candidates = []
    for root in (point_label_dir, segmentation_dir):
        for stem in stems:
            candidates.extend([os.path.join(root, f"{stem}.npy"), os.path.join(root, f"{stem}.npz")])

    for path in candidates:
        if not os.path.isfile(path):
            continue
        labels = _load_numpy_segmentation(path, cache)
        if labels is None:
            continue
        labels = np.asarray(labels)
        if labels.ndim in (1, 2) and num_points in labels.shape:
            print(f"Loaded point-level SAM part labels for obj_{object_id:03d}: {path}")
            return labels
        print(
            f"Warning: ignoring point-level SAM labels {path}; "
            f"shape {labels.shape} does not match {num_points} object points."
        )
    return None


def _project_view_labels_to_object_points(
    args,
    segmentation_dir,
    object_id,
    object_points,
    object_tracklets_vehicle,
    selected_frames,
    raw_intrinsics,
    cam_to_egos,
    camera_list,
    original_sizes,
    cache,
):
    num_points = object_points.shape[0]
    points_h = np.concatenate([object_points, np.ones((num_points, 1))], axis=1)
    votes = defaultdict(lambda: np.zeros(num_points, dtype=np.int32))
    num_views = 0
    start_frame, end_frame = selected_frames

    for frame_offset, frame in enumerate(range(start_frame, end_frame + 1)):
        matching_tracklet = next(
            (
                tracklet
                for tracklet in object_tracklets_vehicle[frame_offset]
                if int(tracklet[0]) == object_id
            ),
            None,
        )
        if matching_tracklet is None:
            continue

        obj_to_ego = np.eye(4)
        obj_to_ego[:3, :3] = quaternion_to_matrix_numpy(matching_tracklet[4:8])
        obj_to_ego[:3, 3] = matching_tracklet[1:4]

        for cam_id in camera_list:
            segmentation_path = os.path.join(segmentation_dir, f"{frame:06d}_{cam_id}.npy")
            if not os.path.isfile(segmentation_path):
                segmentation_path = os.path.join(segmentation_dir, f"{frame:06d}_{cam_id}.npz")
            if not os.path.isfile(segmentation_path):
                continue

            payload = _load_numpy_segmentation(segmentation_path, cache)
            labels = _coerce_view_segmentation_labels(payload, segmentation_path)
            if labels is None:
                continue

            cam_from_obj = np.linalg.inv(cam_to_egos[cam_id]) @ obj_to_ego
            points_camera = points_h @ cam_from_obj.T
            xyz_camera = points_camera[:, :3]
            z = xyz_camera[:, 2]
            valid = z > 1e-6
            if not np.any(valid):
                continue

            height, width = labels.shape
            original_height, original_width = original_sizes[cam_id]
            intrinsic = raw_intrinsics[cam_id]
            fx = intrinsic[0, 0] * width / original_width
            fy = intrinsic[1, 1] * height / original_height
            cx = intrinsic[0, 2] * width / original_width
            cy = intrinsic[1, 2] * height / original_height
            x = np.rint(fx * xyz_camera[:, 0] / np.maximum(z, 1e-6) + cx).astype(np.int64)
            y = np.rint(fy * xyz_camera[:, 1] / np.maximum(z, 1e-6) + cy).astype(np.int64)
            valid &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
            if not np.any(valid):
                continue

            point_indices = np.where(valid)[0]
            sampled_labels = labels[y[valid], x[valid]].astype(np.int64, copy=False)
            foreground = sampled_labels > 0
            point_indices = point_indices[foreground]
            sampled_labels = sampled_labels[foreground]
            if sampled_labels.size == 0:
                continue

            num_views += 1
            for label in np.unique(sampled_labels):
                np.add.at(votes[int(label)], point_indices[sampled_labels == label], 1)

    if len(votes) == 0:
        return None

    min_views = int(max(1, getattr(args, "sam_part_min_views", 1)))
    best_labels = np.full(num_points, -1, dtype=np.int64)
    best_counts = np.zeros(num_points, dtype=np.int32)
    for label in sorted(votes):
        counts = votes[label]
        replace = counts > best_counts
        best_counts[replace] = counts[replace]
        best_labels[replace] = label
    best_labels[best_counts < min_views] = -1

    assigned = int((best_labels >= 0).sum())
    if assigned == 0:
        return None
    print(
        f"Projected SAM part labels for obj_{object_id:03d}: "
        f"{assigned}/{num_points} points from {num_views} labeled views."
    )
    return best_labels


def load_object_sam_part_labels(
    args,
    object_id,
    object_points,
    object_tracklets_vehicle,
    selected_frames,
    raw_intrinsics,
    cam_to_egos,
    camera_list,
    original_sizes,
    cache,
):
    if not _use_seg_hetero_dynamics(args):
        return None

    segmentation_dir = _resolve_sam_part_segmentation_dir(args)
    if not os.path.isdir(segmentation_dir):
        return None

    direct_labels = _load_direct_object_part_labels(
        segmentation_dir,
        object_id,
        object_points.shape[0],
        cache,
    )
    if direct_labels is not None:
        return direct_labels

    return _project_view_labels_to_object_points(
        args=args,
        segmentation_dir=segmentation_dir,
        object_id=object_id,
        object_points=object_points,
        object_tracklets_vehicle=object_tracklets_vehicle,
        selected_frames=selected_frames,
        raw_intrinsics=raw_intrinsics,
        cam_to_egos=cam_to_egos,
        camera_list=camera_list,
        original_sizes=original_sizes,
        cache=cache,
    )


def load_camera_info(datadir):
    ego_pose_dir = os.path.join(datadir, 'ego_pose')
    # extrinsics_dir = os.path.join(datadir, 'extrinsics')
    # intrinsics_dir = os.path.join(datadir, 'intrinsics')

    # intrinsics = []
    # extrinsics = []
    # for i in range(5):
    #     intrinsic = np.loadtxt(os.path.join(intrinsics_dir, f"{i}.txt"))
    #     fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
    #     intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    #     intrinsics.append(intrinsic)

    # for i in range(5):
    #     cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir, f"{i}.txt"))
    #     extrinsics.append(cam_to_ego)

    ego_frame_poses = []
    ego_cam_poses = [[] for i in range(5)]
    ego_pose_paths = sorted(os.listdir(ego_pose_dir))
    for ego_pose_path in ego_pose_paths:

        # frame pose
        if '_' not in ego_pose_path:
            ego_frame_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_frame_poses.append(ego_frame_pose)
        else:
            cam = image_filename_to_cam(ego_pose_path)
            ego_cam_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_cam_poses[cam].append(ego_cam_pose)

    # center ego pose
    ego_frame_poses = np.array(ego_frame_poses)
    center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
    ego_frame_poses[:, :3, 3] -= center_point  # [num_frames, 4, 4], ego -> world

    ego_cam_poses = [np.array(ego_cam_poses[i]) for i in range(5)]  # ego -> world
    ego_cam_poses = np.array(ego_cam_poses)
    ego_cam_poses[:, :, :3, 3] -= center_point  # [5, num_frames, 4, 4]
    return ego_frame_poses, ego_cam_poses  # 自车的世界坐标系位姿，相机的世界坐标系位姿


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def transform_poses_pca(poses, fix_radius=0):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
    
    From https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/af86ea6340b9be6b90ea40f66c0c02484dfc7302/internal/camera_utils.py#L161
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    if fix_radius > 0:
        scale_factor = 1. / fix_radius
    else:
        scale_factor = 1. / (np.max(np.abs(poses_recentered[:, :3, 3])) + 1e-5)
        scale_factor = min(1 / 10, scale_factor)

    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    # PCA 对齐、居中、缩放之后的 相机位姿（camera-to-world）；
    # 原始世界坐标系 → PCA 坐标系 的刚体变换矩阵（包含旋转和平移，以及后续的缩放）；
    # 场景缩放因子，表示原始场景被缩放到 PCA 坐标系的比例
    return poses_recentered, transform, scale_factor


def readDriveXWaymoInfo(args):
    load_dynamic_mask = args.load_dynamic_mask
    load_bbox_mask = args.load_bbox_mask
    neg_fov = args.neg_fov
    start_time = args.start_time  # 第start_time帧
    end_time = args.end_time  # 若为 -1，则设为帧数上限

    ORIGINAL_SIZE = [[1280, 1920], [1280, 1920], [1280, 1920], [884, 1920], [884, 1920]]
    load_size = [640, 960]

    cam_infos = []
    points = []
    points_time = []

    data_root = args.source_path
    image_folder = os.path.join(data_root, "images")
    num_seqs = len(os.listdir(image_folder)) / 5
    if end_time == -1:
        end_time = int(num_seqs)
    else:
        end_time += 1

    frame_num = end_time - start_time  # 定义时间范围，即使用第 start_time 帧至第 end_time-1 帧的内容
    # assert frame_num == 50, "frame_num should be 50"
    time_duration = args.time_duration
    time_interval = (time_duration[1] - time_duration[0]) / (end_time - start_time)  # 将帧编号映射到物理时间 t ∈ [time_duration[0], time_duration[1]] = [0, 1], 每帧之间的时间间隔是 time_interval

    camera_list = args.camera_list
    truncated_min_range, truncated_max_range = 2, 80

    # ---------------------------------------------
    # load poses: intrinsic, c2w, l2w per camera
    # ---------------------------------------------
    _intrinsics = [None] * 5
    _raw_intrinsics = [None] * 5
    _distortions = [None] * 5
    cam_to_egos = [None] * 5
    for i in camera_list:
        # load intrinsics
        intrinsic = np.loadtxt(os.path.join(data_root, "intrinsics", f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        k1, k2, p1, p2, k3 = intrinsic[4], intrinsic[5], intrinsic[6], intrinsic[7], intrinsic[8]
        _raw_intrinsics[i] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        # scale intrinsics w.r.t. load size
        fx, fy = (
            fx * load_size[1] / ORIGINAL_SIZE[i][1],
            fy * load_size[0] / ORIGINAL_SIZE[i][0],
        )
        cx, cy = (
            cx * load_size[1] / ORIGINAL_SIZE[i][1],
            cy * load_size[0] / ORIGINAL_SIZE[i][0],
        )
        _intrinsics[i] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        _distortions[i] = np.array([k1, k2, p1, p2, k3])
        # load extrinsics
        cam_to_ego = np.loadtxt(os.path.join(data_root, "extrinsics", f"{i}.txt"))
        cam_to_egos[i] = cam_to_ego  # opencv_cam -> waymo_cam -> waymo_ego

    ego_frame_poses, ego_cam_poses = load_camera_info(data_root)

    # ---------------------------------------------
    # get c2w and w2c transformation per frame and camera 
    # ---------------------------------------------
    # compute per-image poses and intrinsics
    cam_to_worlds, ego_to_worlds = [], []  # ego_to_worlds: list, len = num_frames, each element: [4, 4]
    lidar_to_worlds = []
    ego_to_world_start = ego_frame_poses[start_time]
    for t in range(start_time, end_time):
        # ego to world transformation: cur_ego -> world -> start_ego(world)
        ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_frame_poses[t]
        ego_to_worlds.append(ego_to_world)
        for cam_id in camera_list:
            # transformation:
            # opencv_cam -> waymo_cam -> waymo_cur_ego -> world -> start_ego(world)
            cam_ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_cam_poses[cam_id, t]
            cam2world = cam_ego_to_world @ cam_to_egos[cam_id]
            cam_to_worlds.append(cam2world)
        # lidar to world : lidar = ego in waymo
        lidar_to_worlds.append(ego_to_world)  # same as ego_to_worlds
    # convert to numpy arrays
    cam_to_worlds = np.stack(cam_to_worlds, axis=0)
    lidar_to_worlds = np.stack(lidar_to_worlds, axis=0)

    # ---------------------------------------------
    # get image, sky_mask, lidar per frame and camera 
    # ---------------------------------------------
    pointcloud_path = os.path.join(data_root, 'pointcloud.npz')  # lidar点云
    pts3d_dict = np.load(pointcloud_path, allow_pickle=True)['pointcloud'].item()  # 3D点云
    # pts2d_dict = np.load(pointcloud_path, allow_pickle=True)['camera_projection'].item()  # 3D点云投影到2D图像

    # object_tracklets_vehicle.shape = [num_frames, num_objects, 8], 8: id(1) + position(3) + quaternion(4)
    # len(object_info) = num_objects, track_id, class, class_label(refer to waymo_utils.py), height, width, length, deformable, start_frame, end_frame(closed interval, contain start_frame and end_frame)
    _, object_tracklets_vehicle, object_info = get_obj_pose_tracking(
        data_root,
        [start_time, end_time - 1],
        ego_frame_poses,
        camera_list,
    )
    ply_path = os.path.join(args.model_path, "input_ply")

    build_pointcloud(args, data_root, object_tracklets_vehicle, object_info,
                         [start_time, end_time - 1], ego_frame_poses, camera_list)

    os.makedirs(os.path.join(args.source_path, "dynamic_mask_select"), exist_ok=True)
    dynamic_bbox_info, bbox_mask_list = build_bbox_mask(args, object_tracklets_vehicle, object_info,
                        [start_time, end_time - 1], _intrinsics, cam_to_egos, camera_list)

    os.makedirs(os.path.join(args.source_path, "refined_bbox"), exist_ok=True)
    for idx, t in enumerate(tqdm(range(start_time, end_time), desc="Loading data", bar_format='{l_bar}{bar:50}{r_bar}')):
        images = []
        image_paths = []
        HWs = []
        sky_masks = []
        dynamic_masks = []
        bbox_masks = []

        for cam_idx in camera_list:
            image_path = os.path.join(args.source_path, "images", f"{t:06d}_{cam_idx}.png")
            im_data = Image.open(image_path)
            im_data = im_data.resize((load_size[1], load_size[0]), Image.BILINEAR)  # PIL resize: (W, H)
            W, H = im_data.size
            image = np.array(im_data) / 255.
            HWs.append((H, W))
            images.append(image)
            image_paths.append(image_path)

            sky_path = os.path.join(args.source_path, "sky_mask", f"{t:06d}_{cam_idx}.png")
            sky_data = Image.open(sky_path)
            sky_data = sky_data.resize((load_size[1], load_size[0]), Image.NEAREST)  # PIL resize: (W, H)
            sky_mask = np.array(sky_data) > 0
            sky_masks.append(sky_mask.astype(np.float32))

            if load_bbox_mask:
                view_idx = t * 10 + cam_idx
                bbox_mask = bbox_mask_list[view_idx]
                bbox_masks.append(bbox_mask.astype(np.float32))

            if load_dynamic_mask:
                view_idx = t * 10 + cam_idx
                dynamic_bbox_list = dynamic_bbox_info[view_idx]

                seg_path = os.path.join(args.source_path, "seg_npy", f"{t:06d}_{cam_idx}.npy")
                seg_data = np.load(seg_path)
                # resize
                seg_data = cv2.resize(seg_data, (load_size[1], load_size[0]), interpolation=cv2.INTER_NEAREST)
                max_seg_value = seg_data.max()
                bbox_dict = defaultdict(list)
                for seg_value in range(1, max_seg_value + 1):
                    seg_mask = (seg_data == seg_value)
                    if seg_mask.sum() == 0:
                        continue
                    ideal_bbox = -1
                    max_overlap = 0
                    for bbox_idx, dynamic_bbox in enumerate(dynamic_bbox_list):
                        overlap = dynamic_bbox[seg_mask].sum()
                        if overlap > 0.5 * seg_mask.sum() and overlap > max_overlap:
                            max_overlap = overlap
                            ideal_bbox = bbox_idx
                    if ideal_bbox != -1:
                        bbox_dict[ideal_bbox].append(seg_mask)

                cur_view_mask = np.zeros_like(seg_data)
                for bbox_idx, seg_masks in bbox_dict.items():
                    # choose max IoU
                    cur_bbox_mask = max(seg_masks, key=lambda x: dynamic_bbox_list[bbox_idx][x].sum() / np.logical_or(dynamic_bbox_list[bbox_idx], x).sum())
                    cur_view_mask = np.logical_or(cur_view_mask, cur_bbox_mask)

                refined_bbox_path = os.path.join(args.source_path, "refined_bbox", f"{t:06d}_{cam_idx}.png")
                Image.fromarray(cur_view_mask.astype(np.uint8) * 255).save(refined_bbox_path)

                dynamic_masks.append(cur_view_mask.astype(np.float32))

        timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * idx / (frame_num - 1)  # 对每一帧，计算该帧在时间区间内的物理时间戳 timestamp

        lidar_points = pts3d_dict[t]
        # select lidar points based on a truncated ego-forward-directional range
        # make sure most of lidar points are within the range of the camera
        valid_mask = lidar_points[:, 0] < truncated_max_range
        valid_mask = valid_mask & (lidar_points[:, 0] > truncated_min_range)
        lidar_points = lidar_points[valid_mask]
        # transform lidar points to world coordinate system
        lidar_points = (
                lidar_to_worlds[idx][:3, :3] @ lidar_points.T
                + lidar_to_worlds[idx][:3, 3:4]
        ).T  # point_xyz_world

        points.append(lidar_points)
        point_time = np.full_like(lidar_points[:, :1], timestamp)
        points_time.append(point_time)

        for cam_idx in camera_list:
            # world-lidar-pts --> camera-pts : w2c
            c2w = cam_to_worlds[int(len(camera_list)) * idx + cam_idx]
            w2c = np.linalg.inv(c2w)
            point_camera = (
                    w2c[:3, :3] @ lidar_points.T
                    + w2c[:3, 3:4]
            ).T

            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            K = _intrinsics[cam_idx]
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            height, width = HWs[cam_idx]
            if neg_fov:
                FovY = -1.0
                FovX = -1.0
            else:
                FovY = focal2fov(fy, height)
                FovX = focal2fov(fx, width)

            if args.undistort:
                image = cv2.undistort(
                    np.array(images[cam_idx]),
                    _intrinsics[cam_idx],
                    _distortions[cam_idx],
                )
                sky_mask = cv2.undistort(
                    np.array(sky_masks[cam_idx]),
                    _intrinsics[cam_idx],
                    _distortions[cam_idx],
                )
                dynamic_mask = cv2.undistort(
                    np.array(dynamic_masks[cam_idx]),
                    _intrinsics[cam_idx],
                    _distortions[cam_idx],
                )
                bbox_mask = cv2.undistort(
                    np.array(bbox_masks[cam_idx]),
                    _intrinsics[cam_idx],
                    _distortions[cam_idx],
                )
            else:
                image = images[cam_idx]
                sky_mask = sky_masks[cam_idx]
                dynamic_mask = dynamic_masks[cam_idx] if load_dynamic_mask else None
                bbox_mask = bbox_masks[cam_idx] if load_bbox_mask else None

            cam_infos.append(CameraInfo(uid=idx * 10 + cam_idx, R=R, T=T, FovY=FovY, FovX=FovX,
                                        image=image,
                                        image_path=image_paths[cam_idx], image_name=f"{t:03d}_{cam_idx}",
                                        width=width, height=height, timestamp=timestamp,
                                        pointcloud_camera=point_camera,
                                        fx=fx, fy=fy, cx=cx, cy=cy,
                                        sky_mask=sky_mask,
                                        dynamic_mask=dynamic_mask,
                                        bbox_mask=bbox_mask))
            
    pointcloud = np.concatenate(points, axis=0)
    pointcloud_timestamp = np.concatenate(points_time, axis=0)
    indices = np.random.choice(pointcloud.shape[0], args.num_pts, replace=True)
    pointcloud = pointcloud[indices]
    pointcloud_timestamp = pointcloud_timestamp[indices]

    w2cs = np.zeros((len(cam_infos), 4, 4))
    Rs = np.stack([c.R for c in cam_infos], axis=0)
    Ts = np.stack([c.T for c in cam_infos], axis=0)
    w2cs[:, :3, :3] = Rs.transpose((0, 2, 1))
    w2cs[:, :3, 3] = Ts
    w2cs[:, 3, 3] = 1
    c2ws = unpad_poses(np.linalg.inv(w2cs))
    # PCA 对齐、居中、缩放之后的 相机位姿（camera-to-world）；
    # 原始世界坐标系 → PCA 坐标系 的刚体变换矩阵（包含旋转和平移，以及后续的缩放）；
    # 场景缩放因子，表示原始场景被缩放到 PCA 坐标系的比例
    c2ws, transform_pca, scale_factor = transform_poses_pca(c2ws, fix_radius=args.fix_radius)
    # if args.static_thresh > 0: # for PVG separation
    #     args.static_thresh = float(args.static_thresh * scale_factor)

    c2ws = pad_poses(c2ws)
    for idx, cam_info in enumerate(tqdm(cam_infos, desc="Transform data", bar_format='{l_bar}{bar:50}{r_bar}')):
        c2w = c2ws[idx]
        w2c = np.linalg.inv(c2w)
        cam_info.R[:] = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        cam_info.T[:] = w2c[:3, 3]
        cam_info.pointcloud_camera[:] *= scale_factor
    pointcloud = (np.pad(pointcloud, ((0, 0), (0, 1)), constant_values=1) @ transform_pca.T)[:, :3]
    if args.eval:
        # for dynamic scene
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num + 1) % args.testhold == 0]

        # for emernerf comparison [testhold::testhold]
        if args.testhold == 10:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold != 0 or (idx // args.cam_num) == 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx // args.cam_num) % args.testhold == 0 and (idx // args.cam_num) > 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization['radius'] = 1 / nerf_normalization['radius']

    pcd = None

    # stage1: read point3d.ply, and initialize as static gaussians
    ply_path = os.path.join(args.model_path, "points3d.ply")
    if not os.path.exists(ply_path):
        rgbs = np.random.random((pointcloud.shape[0], 3))
        storePly(ply_path, pointcloud, rgbs, pointcloud_timestamp)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = BasicPointCloud(pointcloud, colors=np.zeros([pointcloud.shape[0], 3]), normals=None, time=pointcloud_timestamp)

    obj2world_dict = dict()
    obj_timestamp_list = dict()
    for track_id in object_info.keys():
        obj2world_dict[f'obj_{track_id:03d}'] = []
        obj_timestamp_list[f'obj_{track_id:03d}'] = []

    # 静态部分只关注一个pose，即全局场景的起始帧pose，无需构造obj2world_dict
    # 然而动态部分每一帧都有不同的 位置（translation）和朝向（rotation），需要将其初始点云分别变换到 每一帧的世界坐标系
    # 此步骤为每个目标在每帧记录了从目标局部坐标系到世界坐标系的完整变换
    # 该场景/序列有N1个动态目标，每个目标存在于N2帧，因此obj2world_dict最外层的key表示动态目标的编号，内层的value表示该动态目标的位置
    for i, frame in tqdm(enumerate(range(start_time, end_time))):  # 先遍历帧
        # if args.eval:
        #     if (i + 1) % args.testhold == 0:
        #         continue

        ego_pose = ego_frame_poses[frame]  # cur ego -> world
        for tracklet in object_tracklets_vehicle[i]:  # 再遍历每帧的动态目标，最终得到的是各个动态目标在所有帧的信息
            track_id = int(tracklet[0])
            if track_id >= 0:
                obj_pose_vehicle = np.eye(4)
                obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(tracklet[4:8])
                obj_pose_vehicle[:3, 3] = tracklet[1:4]  # object -> ego
                obj_pose_world = ego_pose @ obj_pose_vehicle  # object -> world, [4, 4]
                obj2world_dict[f'obj_{track_id:03d}'].append(obj_pose_world)
                timestamp = time_duration[0] + (time_duration[1] - time_duration[0]) * i / (frame_num - 1)
                obj_timestamp_list[f'obj_{track_id:03d}'].append(timestamp)

    ply_dict = dict()
    ply_dict['bkgd'] = {
        'xyz_array': None, 
        'colors_array': None, 
        'start_frame': start_time, 
        'end_frame': end_time - 1
        }
    for k, v in object_info.items():
        ply_dict[f'obj_{k:03d}'] = {
            'xyz_offset': None, 
            'trajectory': None, 
            'colors_array': None, 
            'sam_part_labels': None,
            'start_frame': v['start_frame'], 
            'end_frame': v['end_frame'], 
            'timestamp_list': obj_timestamp_list[f'obj_{k:03d}']
            }
    # 遍历各个bkgd与object_{}
    sam_segmentation_cache = {}
    for idx, item in enumerate(sorted(os.listdir(os.path.join(args.model_path, "input_ply")))):
        # 先处理静态背景
        # 0 is background
        if idx == 0:
            xyz, colors = load_ply(os.path.join(args.model_path, "input_ply", item))  # 静态点云位置与颜色
            xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1)  # 将xyz增广为齐次坐标
            transform = transform_pca @ np.linalg.inv(ego_to_world_start)  # 将世界坐标转回起始 ego，并对齐主方向
            xyz = xyz @ transform.T
            ply_dict['bkgd']['xyz_array'] = xyz[:, :3]
            ply_dict['bkgd']['colors_array'] = colors
            continue

        # 再处理多个动态前景
        xyz, colors = load_ply(os.path.join(args.model_path, "input_ply", item))  # 动态点云位置与颜色
        obj_idx = int(item.split('_')[2].split('.')[0])

        cur_obj_offset_list = []  # 相对于中心轨迹的位置偏移
        cur_obj_trajectory = []  # 中心轨迹的坐标
        sam_part_labels = load_object_sam_part_labels(
            args=args,
            object_id=obj_idx,
            object_points=xyz,
            object_tracklets_vehicle=object_tracklets_vehicle,
            selected_frames=[start_time, end_time - 1],
            raw_intrinsics=_raw_intrinsics,
            cam_to_egos=cam_to_egos,
            camera_list=camera_list,
            original_sizes=ORIGINAL_SIZE,
            cache=sam_segmentation_cache,
        )
        xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1)
        ply_dict[f'obj_{obj_idx:03d}']['colors_array'] = colors
        ply_dict[f'obj_{obj_idx:03d}']['sam_part_labels'] = sam_part_labels
        for obj2world in obj2world_dict[f'obj_{obj_idx:03d}']:  # 遍历某动态前景在所有帧中的信息，每帧中的动态前景的变换矩阵不同
            # object -> cur ego -> world -> start ego(world)
            transform = np.linalg.inv(ego_to_world_start) @ obj2world  # 将该帧的物体从 world 坐标系变换回以 start 帧 ego 为基准的本地坐标系：（1）统一所有帧的坐标系，便于后续比较与建模；（2）避免全局空间中的位置偏移造成干扰
            transform = transform_pca @ transform  # 引入额外的 PCA 坐标对齐，用于统一方向（可选）
            xyz_world = xyz @ transform.T  # 使用变换后的 obj2world，把每帧的点云从其物体局部坐标系变换到统一世界坐标系。得到每帧的点云在统一世界坐标系下的位置
            xyz_world = xyz_world[:, :3]  # 动态前景表面点坐标，shape：[N, 3]，N表示该序列下某动态前景初始化的点云数量（所有帧加起来的数量）
            # xyz offset(from trajectory)
            trajectory_pos = transform[:3, 3]  # 取出当前物体在统一坐标系下的「中心位置」作为该帧的 trajectory，shape：[3]
            xyz_offset = xyz_world - trajectory_pos  # 计算每个点相对于中心位置的偏移，shape：[N, 3]
            cur_obj_offset_list.append(xyz_offset)  # 最后得到num_frames * [N, 3]的列表，N2表示该动态前景在所有帧中出现的次数
            cur_obj_trajectory.append(trajectory_pos)  # 最后得到num_frames * [3]的列表
        ply_dict[f'obj_{obj_idx:03d}']['xyz_offset'] = np.stack(cur_obj_offset_list, axis=1)  # [N, num_frames, 3]
        ply_dict[f'obj_{obj_idx:03d}']['trajectory'] = np.stack(cur_obj_trajectory, axis=0)  # [num_frames, 3]

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           ply_dict=ply_dict,
                           time_interval=time_interval,
                           time_duration=time_duration,
                           scale_factor=scale_factor)

    return scene_info

# trans (4, 4)
# xyz: (4, N) 
# trans @ xyz = (4, N)

# xyz.T @ trans.T = (trans @ xyz).T = (N, 4) @ (4, 4) = (N, 4)

# start_ego: x front, y right, z down
# opencv: x right, y down, z front
# start_ego->opencv: y->x, x->z, z->y
# save_ply(np.concatenate([ply_82_pca[:,1:2], ply_82_pca[:, 2:3], ply_82_pca[:, 0:1]], axis=1), '082.ply')
