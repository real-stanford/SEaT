
from environment.utils import SCENETYPE, get_scene_volume, get_body_colors, set_visible, get_body_colors
import pybullet as p
import numpy as np
from environment.teleRobotEnv import TeleRobotEnv
from environment.baseEnv import BaseEnv
from learning.dataset import SceneDataset
import hydra
from omegaconf import DictConfig
from pathlib import Path
import shutil
import ray
from utils import depth_to_point_cloud, get_bounds_from_center_pt, get_ray_fn, get_split_file, get_split_obj_roots, seed_all_int, transform_point3s
from utils import init_ray, get_masked_d, get_masked_rgb, ensure_vol_shape, get_device
from utils.tsdfHelper import TSDFHelper, get_single_biggest_cc_single, extend_to_bottom
from utils.pointcloud import sample_pointcloud_from_tsdf
from utils.find_symmetry import get_symmetry_planes, eval_symmetry
from utils.ravenutils import np_unknown_cat
from environment.camera import SimCameraPosition
import random
import h5py
import torch
from tqdm import tqdm 
from os.path import exists

def gen_masked_vol(mask, rgb, d, camera, view_bounds, voxel_size):
    masked_d, masked_rgb = get_masked_d(mask, d), get_masked_rgb(mask, rgb)
    # Create the volume from masked rgb d
    masked_vol_single_image = TSDFHelper.tsdf_from_camera_data(
        views=[(masked_rgb, masked_d, camera.intrinsics, camera.pose_matrix)],
        bounds=view_bounds,
        voxel_size=voxel_size
    )
    return masked_vol_single_image

def get_point_cloud_from_depth(d, camera_intr, camera_pose, view_bounds):
    cam_pts = np.array(depth_to_point_cloud(camera_intr, d))
    world_pts = transform_point3s(camera_pose, cam_pts)
    inside_view_bounds = (
        (world_pts[:, 0] >= view_bounds[0, 0]) & (world_pts[:, 0] < view_bounds[0, 1]) &\
        (world_pts[:, 1] >= view_bounds[1, 0]) & (world_pts[:, 1] < view_bounds[1, 1]) &\
        (world_pts[:, 2] >= view_bounds[2, 0]) & (world_pts[:, 2] < view_bounds[2, 1])
    )
    world_pts = world_pts[inside_view_bounds]
    return world_pts

def get_center_pt_from_d(d, camera_intr, camera_pose, view_bounds):
    world_pts = get_point_cloud_from_depth(d, camera_intr, camera_pose, view_bounds)
    center_pt = None
    if len(world_pts) != 0:
        center_pt = (world_pts.max(axis=0) + world_pts.min(axis=0)) / 2
    return center_pt

def gen_vol_from_mask(mask, d, camera_intr, camera_pose, view_bounds, voxel_size, hw, vol_size):
    masked_d = get_masked_d(mask, d)

    center_pt = get_center_pt_from_d(masked_d, camera_intr, camera_pose, view_bounds)
    if center_pt is None:
        # Object is outside view bounds
        # - we should ignore such object for shape completion input
        # - Put -1 as indices
        cropped_vol_indices = -1 * np.ones((3, 2), dtype=np.int)
    else:
        # - translate the point back to matrix index
        center_indices = np.floor((center_pt - view_bounds[:, 0]) / voxel_size).astype(np.int)
        # - shift center such that cropped volume is inside old volume
        for i in range(3):
            if center_indices[i] - hw[i] < 0:
                center_indices[i] = hw[i] 
            elif center_indices[i] + hw[i] >= vol_size[i]:
                center_indices[i] = vol_size[i] - hw[i] - 1
        # Crop the original volume around the center indices
        cropped_vol_indices = np.array([
            [center_indices[0] - hw[0], center_indices[0] + hw[0]],
            [center_indices[1] - hw[1], center_indices[1] + hw[1]],
            [center_indices[2] - hw[2], center_indices[2] + hw[2]]
        ], dtype=np.int)
    return cropped_vol_indices, center_pt

def process_mask(mask, obj_body_id_labels):
    # obj_body_id_labels.keys(): set of all objs and kits in the scene
    # obj_body_ids: obj / kits present in the mask image
    obj_body_ids = set(obj_body_id_labels.keys()).intersection(np.unique(mask))
    num_objs = len(obj_body_ids)
    masks = np.zeros((num_objs, *mask.shape), dtype=np.uint8)
    labels = np.empty(num_objs)
    boxes = np.empty((num_objs, 4), dtype=float)
    # sometimes object is dropped into the scene, but it slids away from the scene.
    # we need to discard such objects from data
    valid_indices = np.ones((num_objs), dtype=bool) 
    for idx, obj_body_id in enumerate(obj_body_ids):
        pos = np.where(mask == obj_body_id)
        if len(pos[0]) == 0 or len(pos[1]) == 0:
            valid_indices[idx] = False
            continue
        masks[idx][mask == obj_body_id] = 1
        xmin, xmax = np.min(pos[1]), np.max(pos[1])
        ymin, ymax = np.min(pos[0]), np.max(pos[0])
        if xmax - xmin <= 0 or ymax - ymin <= 0:
            valid_indices[idx] = False
            continue
        boxes[idx] = np.array([xmin, ymin, xmax, ymax])
        labels[idx] = obj_body_id_labels[obj_body_id] 
    return masks, labels, boxes, valid_indices, obj_body_ids

def generate_dataset(output_dir: Path, cfg:DictConfig, obj_paths:list):
    env = TeleRobotEnv(cfg, obj_paths=obj_paths, dataset_split=cfg.data_gen.dataset_split)
    stype = cfg.data_gen.scene_type
    if stype == 'kit':
        scene_prob_dist = np.array([1,0,0])
        cropped_vol_shape = np.array(cfg.env.kit_vol_shape)
    elif stype == 'object':
        scene_prob_dist = np.array([0,1,0])
        cropped_vol_shape = np.array(cfg.env.obj_vol_shape)
    else:
        # scene_prob_dist = np.array([0,0,1])
        print(f'Scene type not handled: {stype}')
        raise NotImplementedError
    obj_body_id_labels, scene_type = env.reset_scene_random(output_dir, scene_prob_dist=scene_prob_dist, six_dof=True)
    # capture scene volume
    # - we don't need images from bottom camera, since the plane is visible
    gt_cameras, view_bounds = env.get_gt_cameras(scene_type), env.get_view_bounds(scene_type) 
    camera = env.get_camera(scene_type)
    rgb, d, mask = camera.get_image(seg_mask=True)

    masks, labels, boxes, valid_indices, obj_body_ids = process_mask(mask, obj_body_id_labels)
    if valid_indices.sum() == 0:
        print("No object / kit present in scene. Continuing")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        return

    # generate volumes for instance shape completion:
    # - hide all
    visual_data = dict()
    visual_data[env.plane_id] = get_body_colors(env.plane_id)
    for obj_body_id in obj_body_id_labels:
        visual_data[obj_body_id] = get_body_colors(obj_body_id)
    set_visible(visual_data, visible=False)
    env.ur5.set_visible(visible=False)

    # Required changes:
    # - For every object:
    kit_indices = np.zeros_like(valid_indices)
    voxel_size = cfg.env.voxel_size
    sc_inps = np.empty((0, *(cropped_vol_shape)))
    sc_targets = np.empty((0, *(cropped_vol_shape)))
    for idx, obj_body_id in enumerate(obj_body_ids):
        if not valid_indices[idx]:
            continue
        if stype != 'kit' and obj_body_id_labels[obj_body_id] == env.label_kit:
            kit_indices[idx] = True
            continue
    #   * Hide everything except the object
        set_visible({obj_body_id: visual_data[obj_body_id]}, visible=True)
    #   * calculate bounds using the mask
    #       - For this, first mask out the depth image.
        masked_d = get_masked_d(masks[idx], d)
    #       - Then get the center pt for mask
        center_pt = get_center_pt_from_d(masked_d, camera.intrinsics, camera.pose_matrix, view_bounds)
        if center_pt is None:
            print("center pt for one of the object is outside view bounds. Ignorning the datapoint")
            continue
    #       - Create bounds around center_pt
    #       - Adjust bounds
        crop_bounds = get_bounds_from_center_pt(center_pt, cropped_vol_shape, voxel_size, view_bounds)
    #   * Mask out primary camera depth and generate volume using crop_bounds and original (occluded) rgb d
        sc_inp = TSDFHelper.tsdf_from_camera_data([(rgb, masked_d, camera.intrinsics, camera.pose_matrix)], crop_bounds, voxel_size)
        sc_inp = ensure_vol_shape(sc_inp, cropped_vol_shape)
        if (sc_inp.shape != cropped_vol_shape).any():
            print("Found sc_inp shape mismatch. Ignoring the datapoint")
            continue
        sc_inps = np_unknown_cat(sc_inps, sc_inp)
    #   * Capture depth images in the scene where only the object is visible
    #   * Generate gt volume using these depth images and crop_bounds
        sc_target, _, _, _ = get_scene_volume(gt_cameras, crop_bounds, voxel_size)
        sc_targets = np_unknown_cat(sc_targets, sc_target)
        set_visible({obj_body_id: visual_data[obj_body_id]}, visible=False)
    if len(sc_inps) == 0:
        print("length of sc_inps 0. Ignorning the whole episode")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        return
    masks = masks[valid_indices]
    boxes = boxes[valid_indices]
    labels = labels[valid_indices]

    SceneDataset.extend(
        rgb, d, masks, labels, boxes,
        sc_inps, sc_targets,
        output_dir
    )

def save_vol_match_sample(rgbd_img, vol_pos_info, snap_data, tn_data, d2o_data, obj_info, output_dir):
    rgb, d, mask = rgbd_img
    all_vol, all_pos, raw_kit_cnt = vol_pos_info
    obj_vol, raw_obj_vol, kit_vol, raw_kit_vol, p1_coords, p1_ori = snap_data
    cmap, hmap, pick_pos, place_pos, place_ori = tn_data
    obj_img, kit_img, pc, ori = d2o_data
    concav_ori, symmetry, obj_id = obj_info

    output_dir.mkdir(exist_ok=True, parents=True)
    with h5py.File(str(output_dir / "data.hdf"), "w") as hdf:
        hdf.create_dataset("rgb", data=rgb)
        hdf.create_dataset("d", data=d)
        hdf.create_dataset("mask", data=mask)

        hdf.create_dataset("all_vol", data=all_vol)
        hdf.create_dataset("all_pos", data=all_pos)
        hdf.create_dataset("kit_pos", data=raw_kit_cnt)

        hdf.create_dataset("p0_vol", data=obj_vol)
        hdf.create_dataset("p0_vol_raw", data=raw_obj_vol)
        hdf.create_dataset("p1_vol", data=kit_vol)
        hdf.create_dataset("p1_vol_raw", data=raw_kit_vol)
        hdf.create_dataset("p1_coords", data=p1_coords)
        hdf.create_dataset("p1_ori", data=p1_ori)

        hdf.create_dataset("cmap", data=cmap)
        hdf.create_dataset("hmap", data=hmap)
        hdf.create_dataset("pick_pos", data=pick_pos)
        hdf.create_dataset("place_pos", data=place_pos)
        hdf.create_dataset("place_ori", data=place_ori)

        hdf.create_dataset("part_img", data=obj_img)
        hdf.create_dataset("kit_img", data=kit_img)
        hdf.create_dataset("pc", data=pc)
        hdf.create_dataset("ori", data=ori)

        hdf.create_dataset("concav_ori", data=concav_ori)
        hdf.create_dataset("symmetry", data=symmetry)
        hdf.create_dataset("obj_id", data=obj_id)

def gen_raw_volume(camera, view_bounds, obj_id, vol_shape, voxel_size):
    """subroutine to generate raw volumes"""
    rgb, d, mask = camera.get_image(seg_mask=True)
    processed_mask = np.zeros_like(mask)
    processed_mask[mask==obj_id] = 1
    if np.sum(processed_mask) == 0:
        print('The object is not visible in the mask')
        return None
    masked_d = get_masked_d(processed_mask, d)
    center_pt = get_center_pt_from_d(masked_d, camera.intrinsics, camera.pose_matrix, view_bounds)
    if center_pt is None:
        print("The center point of the object is outside view bounds.")
        return None
    crop_bounds = get_bounds_from_center_pt(center_pt, vol_shape, voxel_size, view_bounds)
    volume = TSDFHelper.tsdf_from_camera_data([(rgb, masked_d, camera.intrinsics, camera.pose_matrix)], crop_bounds, voxel_size*1.0001)
    volume = ensure_vol_shape(volume, vol_shape)
    if (volume.shape != vol_shape).any():
        print('The generated volume has invalid shape: ', volume.shape)
        return None
    center_pt = (crop_bounds[:,1] - crop_bounds[:,0]) / 2 + crop_bounds[:,0]
    return volume, center_pt, crop_bounds
 
def generate_dataset_vol_match_6DoF(output_dir: Path, cfg: DictConfig):
    # get configs and init dir
    voxel_size = cfg.env.voxel_size
    look_at = np.array(cfg.env.look_at)
    view_bounds = np.array(cfg.env.view_bounds)
    obj_view_bnds = np.array(cfg.env.view_bounds_objects)
    kit_view_bnds = np.array(cfg.env.view_bounds_kit)
    dataset_split = cfg.vol_match_6DoF.dataset_split
    p0_vol_shape = np.array(cfg.vol_match_6DoF.p0_vol_shape_gen)
    p1_vol_shape = np.array(cfg.vol_match_6DoF.p1_vol_shape_gen)
    image_size = np.array(cfg.vol_match_6DoF.image_size)
    image_size_d2o = np.array(cfg.vol_match_6DoF.image_size_d2o)
    pix_size = (view_bounds[0, 1] - view_bounds[0, 0]) / image_size[0]
    output_dir.mkdir(exist_ok=True)

    success = False
    while not success:
        # reset env
        env = TeleRobotEnv(cfg, dataset_split=dataset_split, gui=False)
        gt_cam = env.get_gt_camera()
        obj_cam = env.get_camera(SCENETYPE.OBJECTS)
        kit_cam = env.get_camera(SCENETYPE.KIT)
        _, obj_details, kit_body_id = env.reset_scene_6DoF(output_dir, unit_kit=False, load_single_object=False)

        rgbd_img = gt_cam.get_image(seg_mask=True)
        all_vol, all_pos = [], []
        for obj_detail in obj_details:
            obj_body_id = obj_detail["body_id"]
            raw_obj_vol_data = gen_raw_volume(obj_cam, obj_view_bnds, obj_body_id, p0_vol_shape, voxel_size)
            if raw_obj_vol_data is None:
                continue
            raw_obj_vol, raw_obj_cnt, obj_crop_bounds = raw_obj_vol_data
            all_vol.append(raw_obj_vol)
            all_pos.append(raw_obj_cnt)
        all_vol, all_pos = np.array(all_vol), np.array(all_pos)

        # randomly choose an object as targett
        obj_id = random.randint(0, len(obj_details)-1)
        obj_detail = obj_details[obj_id]
        obj_body_id = obj_detail["body_id"]

        # pose details
        obj_kit_frame = obj_detail["position"], p.getQuaternionFromEuler(obj_detail["orientation"])
        kit_pos, kit_ori = p.getBasePositionAndOrientation(kit_body_id)  # kit in world frame
        obj_pos, obj_ori = p.getBasePositionAndOrientation(obj_body_id)
        kit_pos, obj_pos = np.array(kit_pos), np.array(obj_pos)
        concav_pos, concav_ori = p.multiplyTransforms(kit_pos, kit_ori, *obj_kit_frame)

        # transformations when both volumes are placed with their centers aligned with origin
        T_obj_to_world_ori = p.invertTransform((0, 0, 0), obj_ori)
        T_obj_to_concav = p.multiplyTransforms(*obj_kit_frame, *T_obj_to_world_ori) # obj rotated at origin, and kit at origin
        obj_kit_vol_frame_T_obj_kit_can_frame = p.multiplyTransforms([0, 0, 0], kit_ori, *T_obj_to_concav) # obj rotated at origin, kit rotated at origin
        place_ori = obj_kit_vol_frame_T_obj_kit_can_frame[1] # for depth2orient and transporternet
        place_pos = np.array(obj_kit_vol_frame_T_obj_kit_can_frame[0]) + kit_pos # for transporternet

        # visual data
        visual_data_plane = {env.plane_id: get_body_colors(env.plane_id)}
        visual_data_obj_all = {
            obi["body_id"]: get_body_colors(obi["body_id"]) for obi in obj_details
        }
        visual_data_obj = {obj_body_id: get_body_colors(obj_body_id)}
        visual_data_kit = {kit_body_id: get_body_colors(kit_body_id)}

        # Hide robot
        env.ur5.set_visible(False)

        # get transporternet data
        def in_bounds(p, vb):
            return vb[0][0] <= p[0] and vb[0][1] >= p[0] and \
                vb[1][0] <= p[1] and vb[1][1] >= p[1]
        if not (in_bounds(obj_pos, obj_view_bnds) and in_bounds(place_pos, kit_view_bnds)):
            print("Out of TN view bounds.")
            continue
        front_cam = SimCameraPosition([1.5, 0,   1], look_at, [0,0,1], image_size=image_size)
        right_cam = SimCameraPosition([0.3, 1.5, 1], look_at, [0,0,1], image_size=image_size)
        left_cam = SimCameraPosition([0.3, -1.5, 1], look_at, [0,0,1], image_size=image_size)
        cmap_tn, hmap_tn = BaseEnv.get_scene_cmap_hmap([front_cam, right_cam, left_cam], view_bounds, pix_size)

        # Hide everyting
        set_visible({**visual_data_plane, **visual_data_kit, **visual_data_obj_all}, False)

        # Collect Data for SnapNetwork and Depth2Orient
        # Object volume
        set_visible(visual_data_obj, True)
        # - For Snap
        raw_obj_vol_data = gen_raw_volume(obj_cam, obj_view_bnds, obj_body_id, p0_vol_shape, voxel_size)
        if raw_obj_vol_data is None:
            continue
        raw_obj_vol, raw_obj_cnt, obj_crop_bounds = raw_obj_vol_data
        obj_vol, _, _, _ = get_scene_volume(env.gt_cameras_objects, obj_crop_bounds, voxel_size)
        # - For Depth2Orient
        pc_obj = sample_pointcloud_from_tsdf(obj_vol, 2048) # pointcloud for loss calculation
        obj_cam_pos = obj_pos + np.array([0,0,0.3])
        obj_cam_d2o = SimCameraPosition(obj_cam_pos, obj_pos, [-1,0,0], image_size_d2o)
        _, obj_img, _ = obj_cam_d2o.get_image()
        set_visible(visual_data_obj, False)
        # Kit volume
        set_visible(visual_data_kit, True)
        # - For Snap
        raw_kit_vol_data = gen_raw_volume(kit_cam, kit_view_bnds, kit_body_id, p1_vol_shape, voxel_size)
        if raw_kit_vol_data is None:
            continue
        raw_kit_vol, raw_kit_cnt, kit_crop_bounds = raw_kit_vol_data
        kit_vol, _, _, _ = get_scene_volume(env.gt_cameras_kit, kit_crop_bounds, voxel_size)
        # - For Depth2Orient
        kit_cam_pos = concav_pos + np.array([0,0,0.2])
        kit_cam_d2o = SimCameraPosition(kit_cam_pos, concav_pos, [-1,0,0], image_size_d2o)
        _, kit_img, _ = kit_cam_d2o.get_image()
        set_visible(visual_data_kit, False)

        # Snap gt pos and ori 
        T_vol_to_obj = (obj_pos - raw_obj_cnt, obj_ori)
        T_vol_to_kit = (kit_pos - raw_kit_cnt, kit_ori)
        T_obj_to_kit = p.multiplyTransforms(*obj_kit_frame, *p.invertTransform(*T_vol_to_obj))
        T_obj_to_concav = p.multiplyTransforms(*T_vol_to_kit, *T_obj_to_kit)
        p1_pos = T_obj_to_concav[0]
        p1_vol_size = kit_crop_bounds[:,1] - kit_crop_bounds[:,0]
        p1_coords = np.ceil(( p1_pos + p1_vol_size/2 ) / voxel_size) # voxel position in p1_volume
        p1_ori = T_obj_to_concav[1] # orientation

        # Find symmetries of the object
        symmetry, pts = get_symmetry_planes(obj_detail['obj_path'], return_pts=True)
        eval_symmetry(pts, symmetry, output_dir, output_dir.name) # comment out after evaluation
        # get orientation of the kit plane
        concav_ori = p.multiplyTransforms(*T_vol_to_kit, *obj_kit_frame)[1]

        snap_data = obj_vol, raw_obj_vol, kit_vol, raw_kit_vol, p1_coords, p1_ori
        tn_data = cmap_tn, hmap_tn, obj_pos, place_pos, place_ori
        d2o_data = obj_img, kit_img, pc_obj, place_ori
        obj_info = concav_ori, symmetry, obj_id
        vol_pos_info = all_vol, all_pos, raw_kit_cnt
        save_vol_match_sample(rgbd_img, vol_pos_info, snap_data, tn_data, d2o_data, obj_info, output_dir)
        success = True

def get_seg_sc_dataset_path(data_cfg: DictConfig):
    dataset_path = Path(data_cfg.dataset_path) / \
        data_cfg.scene_type / data_cfg.dataset_split
    print("Using seg_sc dataset from path: ", dataset_path)
    return dataset_path

def gen_volumes(datadir, num, sc_kit_model_path, sc_obj_model_path):
    device = get_device()
    sc_kit_model = torch.load(sc_kit_model_path, map_location=device).eval()
    sc_obj_model = torch.load(sc_obj_model_path, map_location=device).eval()
    for ind in tqdm(range(num), dynamic_ncols=True):
        datapath = f'{datadir}/{ind}/data.hdf'
        if not exists(datapath):
            continue
        with h5py.File(datapath, "r+") as hdf:
            p0_vol = np.array(hdf.get("p0_vol_raw"))
            p1_vol = np.array(hdf.get("p1_vol_raw"))

        p0_vol_ten = torch.tensor(p0_vol).to(device).float()
        p0_vol_ten = p0_vol_ten.unsqueeze(0).unsqueeze(0)
        p1_vol_ten = torch.tensor(p1_vol).to(device).float()
        p1_vol_ten = p1_vol_ten.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            p0_vol_sc = sc_obj_model(p0_vol_ten).squeeze().cpu().detach().numpy()
            p0_vol_sc = extend_to_bottom(get_single_biggest_cc_single(p0_vol_sc))
            p1_vol_sc = sc_kit_model(p1_vol_ten).squeeze().cpu().detach().numpy()
            p1_vol_sc = extend_to_bottom(get_single_biggest_cc_single(p1_vol_sc))

        with h5py.File(datapath, "r+") as hdf:
            if "p0_vol_sc" in hdf.keys():
                del hdf["p0_vol_sc"]
            if "p1_vol_sc" in hdf.keys():
                del hdf["p1_vol_sc"]
            hdf.create_dataset("p0_vol_sc", data=p0_vol_sc)
            hdf.create_dataset("p1_vol_sc", data=p1_vol_sc)

def main_with_cfg(cfg: DictConfig):
    use_ray = init_ray(cfg.ray)
    kwargs = dict()
    if cfg.data_gen.name == "sc_volumes":
        gen_volumes(cfg.data_gen.datadir, cfg.data_gen.num, cfg.data_gen.sc_kit_path, cfg.data_gen.sc_obj_path)
        return
    if cfg.data_gen.name == "vol_match_6DoF":
        dataset_path = Path(cfg.vol_match_6DoF.dataset_path) / cfg.vol_match_6DoF.dataset_split
        dataset_size = cfg.vol_match_6DoF.dataset_size
    elif cfg.data_gen.name == "seg_sc":
        dataset_path = get_seg_sc_dataset_path(cfg.data_gen)
        dataset_size = cfg.data_gen.dataset_size
    else:
        dataset_path = Path(cfg.data_gen)
        dataset_size = cfg.data_gen.dataset_size
    dataset_path.mkdir(parents=True, exist_ok=True)
    tasks = list()
    gpu_frac=None
    if cfg.data_gen.name == "seg_sc":
        gen_data_fn = generate_dataset
        gpu_frac = cfg.data_gen.gpu_frac
        obj_roots = get_split_obj_roots(get_split_file(cfg.data_gen.dataset_split))
        obj_paths = [obj_root / "obj.urdf" for obj_root in obj_roots]
        kwargs["obj_paths"] = obj_paths
    elif cfg.data_gen.name == "vol_match_6DoF":
        gen_data_fn = generate_dataset_vol_match_6DoF
        gpu_frac = cfg.vol_match_6DoF.gpu_frac
    else:
        raise NotImplementedError

    fn = get_ray_fn(cfg.ray, gen_data_fn, gpu_frac=gpu_frac)
    for i in range(dataset_size):
        tasks.append(fn(dataset_path / str(i), cfg, **kwargs))
    if use_ray:
        ray.get(tasks)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    seed_all_int(cfg.seeds.data_gen)
    main_with_cfg(cfg)

if __name__ == "__main__":
    # TODO: Cache kits pt cloud
    main()