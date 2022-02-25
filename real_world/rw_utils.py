import sys
from pathlib import Path
root_path = Path(__file__).parent.parent.absolute()
# print("Adding to python path: ", root_path)
sys.path = [str(root_path)] + sys.path

import time
import numpy as np
from scipy.ndimage import rotate
from matplotlib import pyplot as plt
from pathlib import Path
from utils.ravenutils import np_unknown_cat
from utils import get_masked_rgb, show_overlay_image
from typing import List, Tuple
from pybullet_utils.bullet_client import BulletClient
from environment.meshRendererEnv import MeshRendererEnv

CAM=0

def get_intrinsics(realworld):
    image_size = (720, 1280)
    focal_length = 891.35443 if realworld else 925.26886 
    intrinsics = np.array([
        [focal_length, 0, float(image_size[1]) / 2],
        [0, focal_length, float(image_size[0]) / 2],
        [0, 0, 1]
    ])
    return intrinsics
    
def get_tn_bounds():
    return (
            np.array([[-0.9, -0.2], [-0.7, 0.7], [0, 0.5]]),
            np.array([[-0.9, -0.2], [-0, 0.7], [0, 0.5]]),
            np.array([[-0.9, -0.2], [-0.7, 0], [0, 0.5]]),
        )

def get_workspace_bounds():
    if CAM==0:
        return np.array([
            [-0.4, 0.2],
            [-0.75, -0.45],
            [-0.03, 0.1]
        ])
    return np.array([
        [-0.6785, -0.3215],
        [-0.357, 0.357],
        [0.015, 0.372]
    ])

def get_client_frame_pose():
    """
    Rough position of camera in robot frame.
    - We will apply inverse of this transform to everything before rendering
        on client
    - Thus, the initial scene the client sees will be 
        as if they are watching from the camera.
    """
    # pos_client = np.array([-0.38, 0.0, -0.18])
    # ori_client = np.array([0, 0, np.pi])
    #print("=======>FIXME<======= using identity client frame pose")
    pos_client = np.zeros((3,))
    ori_client = np.zeros((3,))
    return pos_client, ori_client


def get_tool_init():
    tool_offset = [0.0, -0.1, 0.270 + 0.02079, 0, np.pi, 0]
    tool_orientation = np.array([0, 0,  0], dtype=np.float)
    return tool_offset, tool_orientation


def transform_world_to_camera(point, cam_pose, cam_intr):
    """
        point: (3,)
    """
    point = point.reshape((4, 1)) 
    cam_frame__point = np.linalg.inv(cam_pose) @ point
    cam_frame__point[:3] /= cam_frame__point[3]
    cam_frame__point = cam_frame__point[:3]

    uvw = cam_intr @ cam_frame__point
    u = uvw[0] / uvw[2]
    v = uvw[1] / uvw[2]
    return u[0], v[0]


def transform_world_to_camera_multi(points, cam_pose, cam_intr):
    points_uv = np.empty((0, 2))
    for p_xyz in points:
        u, v = transform_world_to_camera(p_xyz, cam_pose, cam_intr)
        points_uv = np.concatenate((points_uv, np.array([[u, v]])), axis=0)
    return points_uv


def get_crops(color_im, points_uv):
    # Add appropriate padding around original color image (to make it crop safe)
    center = np.ceil((points_uv[0] + points_uv[2]) / 2).astype(np.int)
    diag1 = np.linalg.norm(points_uv[0] - points_uv[2])
    diag2 = np.linalg.norm(points_uv[1] - points_uv[3])
    half_diag = np.ceil(max(diag1, diag2) / 2).astype(np.int)
    pad_color_im = np.pad(color_im, ((half_diag, half_diag), (half_diag, half_diag), (0, 0)))
    # Make a square crop of size diagonal centered around the center of provided bounds
    crop = pad_color_im[center[1]: center[1] + 2 * half_diag, center[0]: center[0] + 2 * half_diag, :]
    # Transform the points to be in cropped image
    points_uv_pad = points_uv + half_diag
    points_uv_crop = points_uv_pad
    points_uv_crop[:, 0] -= center[0]
    points_uv_crop[:, 1] -= center[1]
    # Rotate the cropped image, such that the bounds are axis aligned
    d = points_uv[1] - points_uv[0]
    angle = np.arctan2(d[1], d[0])
    rotate_crop = rotate(crop, angle * 180 / np.pi, reshape=False)
    # Transform points to be in rotated image
    points_uv_crop_center = points_uv_crop - np.array(crop.shape[:2]) / 2
    points_rotated_center = np.empty((0, 2))
    for p in points_uv_crop_center:
        r = np.linalg.norm(p)
        theta = np.arctan2(p[1], p[0])
        new_theta = theta - angle
        new_p = np.array([r * np.cos(new_theta), r*np.sin(new_theta)])
        points_rotated_center = np.concatenate((points_rotated_center, [new_p]), axis=0)
    points_rotated = np.ceil(points_rotated_center + np.array(crop.shape[:2]) / 2).astype(np.int)

    u_min, u_max = np.min(points_rotated[:, 0]), np.max(points_rotated[:, 0])
    v_min, v_max = np.min(points_rotated[:, 1]), np.max(points_rotated[:, 1])
    final_crop = rotate_crop[u_min: u_max, v_min: v_max, :]
    return final_crop, center, angle

def get_bounds_extremes(wb):
    x_min, x_max = wb[0, :]
    y_min, y_max = wb[1, :]
    z_min, z_max = wb[2, :]
    x_mid, y_mid, z_mid  = (wb[:, 0] + wb[:, 1]) / 2
    return x_min, y_min, z_min, x_mid, y_mid, z_mid, x_max, y_max, z_max

    
def get_obj_points(wb=None):
    if wb is None:
        wb = get_workspace_bounds()
    x_min, y_min, z_min, x_mid, y_mid, z_mid, x_max, y_max, z_max = get_bounds_extremes(wb)
    if CAM==0:
        return np.array([
            [x_min, y_min, z_min, 1],
            [x_min, y_max, z_min, 1],
            [x_mid, y_max, z_min, 1],
            [x_mid, y_min, z_min, 1],
        ])
    return np.array([
        [x_min, y_mid, z_min, 1],
        [x_min, y_max, z_min, 1],
        [x_max, y_max, z_min, 1],
        [x_max, y_mid, z_min, 1],
    ])

def get_obj_bounds(wb=None):
    if wb is None:
        wb = get_workspace_bounds()
    x_min, y_min, z_min, x_mid, y_mid, z_mid, x_max, y_max, z_max = get_bounds_extremes(wb)
    if CAM==0:
        return np.array([
            [x_min, x_mid],
            [y_min, y_max],
            [z_min, z_max],
        ])
    return np.array([
        [x_min, x_max],
        [y_mid, y_max],
        [z_min, z_max],
    ])

def get_kit_points(wb=None):
    if wb is None:
        wb = get_workspace_bounds()
    x_min, y_min, z_min, x_mid, y_mid, z_mid, x_max, y_max, z_max = get_bounds_extremes(wb)
    if CAM==0:
        return np.array([
            [x_mid, y_min, z_min, 1],
            [x_mid, y_max, z_min, 1],
            [x_max, y_max, z_min, 1],
            [x_max, y_min, z_min, 1],
        ])
    return np.array([
        [x_min, y_min, z_min, 1],
        [x_min, y_mid, z_min, 1],
        [x_max, y_mid, z_min, 1],
        [x_max, y_min, z_min, 1],
    ])

def get_kit_bounds(wb=None):
    if wb is None:
        wb = get_workspace_bounds()
    x_min, y_min, z_min, x_mid, y_mid, z_mid, x_max, y_max, z_max = get_bounds_extremes(wb)
    if CAM==0:
        return np.array([
            [x_mid, x_max],
            [y_min, y_max],
            [z_min, z_max],
        ])
    return np.array([
        [x_min, x_max],
        [y_min, y_mid],
        [z_min, z_max],
    ])


def get_crops_wb(color_im, depth_im, cam_intr, cam_pose, wb=None):
    if wb is None:
        wb = get_workspace_bounds()

    points_objects = get_obj_points()
    points_kit = get_kit_points()
    points_objects_uv = transform_world_to_camera_multi(points_objects, cam_pose, cam_intr)
    points_kit_uv = transform_world_to_camera_multi(points_kit, cam_pose, cam_intr)
    obj_crop, obj_center, obj_angle = get_crops(color_im, points_objects_uv)
    kit_crop, kit_center, kit_angle = get_crops(color_im, points_kit_uv)
    obj_crop_depth, _, _ = get_crops(np.expand_dims(depth_im, axis=-1), points_objects_uv)
    obj_crop_depth = obj_crop_depth[:, :, 0]
    kit_crop_depth, _, _ = get_crops(np.expand_dims(depth_im, axis=-1), points_kit_uv)
    kit_crop_depth = kit_crop_depth[:, :, 0]
    return obj_crop, kit_crop, obj_crop_depth, kit_crop_depth, obj_center, kit_center, obj_angle, kit_angle

def transform_mask(mask, orig_d, rotate_center, rotate_angle):
    # Given a mask in rotated image (obtained by rotating orig_d about rotate_center by rotate_angle)
    # return the mask in original depth image
    rotate_mask = rotate(mask, -1 * rotate_angle * 180 / np.pi, reshape=False)
    transformed_mask = np.zeros_like(orig_d)
    hh, hw = np.floor(np.array(rotate_mask.shape) / 2).astype(np.int)
    transformed_mask[rotate_center[1] - hh: rotate_center[1] + hh,
                     rotate_center[0] - hw: rotate_center[0] + hw] = rotate_mask[:2*hh, :2*hw]
    return transformed_mask 

def get_obj_masks(bin_cam, cfg, srg):
    rgb, d = bin_cam.get_camera_data(avg_depth=True)
    obj_crop, _, obj_crop_depth, _, obj_center, _, obj_angle, _ = get_crops_wb(
        rgb, d, bin_cam.color_intr, bin_cam.pose, wb=None)

    masks, _, scores = srg.get_seg(obj_crop, obj_crop_depth)
    mask_score_threshold = cfg.perception.seg.mask_score_threshold
    mask_threshold = cfg.perception.seg.mask_threshold

    transformed_masks = None
    for mask, score in zip(masks, scores):
        if score < mask_score_threshold:
            continue
        mask[mask > mask_threshold] = 1
        mask[mask <= mask_threshold] = 0
        transformed_mask = transform_mask(mask, d, obj_center, obj_angle)
        transformed_masks =np_unknown_cat(transformed_masks, transformed_mask)

    return transformed_masks

def get_quadrilateral_mask(orig_image_shape, quadrilateral_points):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    def point_in_triangle(pt, v1, v2, v3):
        d1 = sign(pt, v1, v2)
        d2 = sign(pt, v2, v3)
        d3 = sign(pt, v3, v1)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not(has_neg and has_pos)

    p0, p1, p2, p3 = quadrilateral_points
    mask = np.zeros(orig_image_shape)
    for v in range(mask.shape[0]):
        for u in range(mask.shape[1]):
            if point_in_triangle((u, v), p0, p1, p2) or point_in_triangle((u, v), p2, p3, p0):
                mask[v, u] = 1
    return mask


def get_obj_bounds_mask(cam_pose, cam_intr, orig_img_shape, use_cache=True):
    obj_mask_path =  Path("real_world/obj_mask.npy")
    if use_cache and obj_mask_path.exists():
        print(f"Using cached mask from path: {obj_mask_path}")
        mask = np.load(obj_mask_path)
    else:
        print(f"obj bounds mask cache not found. regenerating ...")
        points_objects = get_obj_points()
        points_objects_uv = transform_world_to_camera_multi(points_objects, cam_pose, cam_intr)
        mask = get_quadrilateral_mask(orig_img_shape, points_objects_uv)
        np.save(obj_mask_path, mask)
    return mask


def get_kit_bounds_mask(cam_pose, cam_intr, orig_img_shape, use_cache=True, show_mask:bool = False):
    kit_mask_path =  Path("real_world/kit_mask.npy")
    if use_cache and kit_mask_path.exists():
        print(f"Using cached mask from path: {kit_mask_path}")
        mask = np.load(kit_mask_path)
    else:
        print(f"kit bounds mask cache not found. regenerating ...")
        points_kit = get_kit_points()
        points_kit_uv = transform_world_to_camera_multi(points_kit, cam_pose, cam_intr)
        # print(points_kit_uv)
        mask = get_quadrilateral_mask(orig_img_shape, points_kit_uv)
        np.save(kit_mask_path, mask)
    return mask

def get_obj_masks_tilted(rgb, d, cam_pose, cam_color_intr, cfg, srg):
    """
        A hacky code that works for non-top-down cameras. Idea is to get the
         mask for the entire image. And ignore the masks that lie
         outside the view bounds.
    """
    obj_mask = get_obj_bounds_mask(cam_pose, cam_color_intr, d.shape)
    # show_overlay_image(obj_mask, rgb)
    masks, _, scores = srg.get_seg(rgb, d)
    mask_score_threshold = cfg.perception.seg.mask_score_threshold
    mask_threshold = cfg.perception.seg.mask_threshold

    tms = None
    tss = None
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # print(score, mask_score_threshold)
        if score < mask_score_threshold:
            continue
        tm =  mask * obj_mask
        if np.any(tm > mask_threshold):
            tms = np_unknown_cat(tms, tm)
            tss = np_unknown_cat(tss, score)
    return tms


def ensure_minus_pi_to_pi(angle_rad):
    while angle_rad < -np.pi:
        angle_rad += 2 * np.pi
    while np.pi < angle_rad:
        angle_rad -= 2 * np.pi
    return angle_rad

def color_mask_rgb(mask, rgb):
    """
    return rgb value for mask (between [0, 1])
    """
    masked_rgb = get_masked_rgb(mask, rgb)
    return masked_rgb.sum(axis=(0, 1)) / (max(1, mask.sum()) * 255)

def fix_ur5_rotation(rpy):
    """
    For some reason, the tool axis are not aligned with robot world axis.
    This function is a result of trial and error.
    """
    roll, pitch, yaw = rpy
    return [-pitch, roll, yaw]

def clip_angle(in_angle, min_angle, max_angle):
    in_angle = ensure_minus_pi_to_pi(in_angle)
    return np.clip(in_angle, min_angle, max_angle)


def get_kit_crop_bounds(bounds_kit: np.ndarray, kit_vol_size: np.ndarray) -> np.ndarray:
    kit_crop_bounds = np.empty((3,2))
    bounds_kit_center = bounds_kit.mean(axis=1) 
    kit_crop_center = np.empty(3)
    kit_crop_center[:2] = bounds_kit_center[:2]
    kit_crop_center[2] = bounds_kit[2, 0] + kit_vol_size[2] / 2
    kit_crop_bounds[:, 0] = kit_crop_center - kit_vol_size / 2
    kit_crop_bounds[:, 1] = kit_crop_center + kit_vol_size / 2
    return kit_crop_bounds

def get_empty_depth():
    return np.load('real_world/kit_bounds_mask_d.npy')

def load_mesh_old_urdf(mesh_data, pbc: BulletClient) -> int:
    """Use when urdf is already dumped"""
    pos = np.array(mesh_data["gt_pos"])
    ori = np.array(mesh_data["gt_ori"])
    mesh_path = Path(mesh_data["mesh_path"])
    urdf_path = MeshRendererEnv.dump_obj_urdf(mesh_path)
    mesh_id = pbc.loadURDF(
        str(urdf_path), basePosition=pos, baseOrientation=ori)
    return mesh_id


def load_mesh(
    pbc: BulletClient,
    mesh_path: Path,
    pos: np.ndarray,
    ori: np.ndarray,
    rgba: np.ndarray = np.ones((4,)),
) -> Path:
    """dump fresh urdf and load mesh"""
    urdf_path = MeshRendererEnv.dump_obj_urdf(mesh_path, rgba=rgba)
    pbc_id = pbc.loadURDF(
        str(urdf_path), basePosition=pos, baseOrientation=ori)
    return pbc_id 


def step_sim(steps: float, pbc: BulletClient) -> None:
    print(f"stepping sim for {steps} steps ...")
    for _ in range(int(steps)):
        pbc.stepSimulation()
        time.sleep(1 / 256)


def remove_pbc_ids(pbc: BulletClient, pbc_ids: List[int]) -> None:
    for pbc_id in pbc_ids:
        pbc.removeBody(pbc_id)
    return list()


def associate_pred_masks_to_gt_masks(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray]
) -> List[Tuple[int, float]]:
    """To associate a pred mask with one of the gt mask: we choose maximum overlap."""
    pred_masks = np.stack(pred_masks)
    gt_masks = np.stack(gt_masks)
    gt_mask_to_pred_mask_matching = [
        (None, -np.inf) for _ in range(len(gt_masks))]
    for i, pred_mask in enumerate(pred_masks):
        overlap_scores = (pred_mask * gt_masks).mean(axis=(1, 2))
        match_index = np.argmax(overlap_scores)
        match_score = overlap_scores[match_index]
        if (
            gt_mask_to_pred_mask_matching[match_index][0] is None or  # didn't matched with anyone yet
            gt_mask_to_pred_mask_matching[match_index][1] < match_score  # previous matching score was lower
        ):
            gt_mask_to_pred_mask_matching[match_index] = (i, match_score)
    return gt_mask_to_pred_mask_matching
