"""
Generate gt data from saved gt_labels.json
Also run evaluation of models along side
"""

from operator import gt
import sys
from pathlib import Path
root_path = Path(__file__).parent.parent.absolute()
# print("Adding to python path: ", root_path)
sys.path = [str(root_path)] + sys.path
import h5py
from environment.meshRendererEnv import dump_vol_render_gif, MeshRendererEnv
from omegaconf import DictConfig
import pybullet
from pybullet_utils.bullet_client import BulletClient
import numpy as np
import hydra
import json
import torch
from utils import (
    ensure_makedirs,
    get_bounds_from_center_pt,
    get_device,
    get_masked_d,
    save_img,
    pad_crop_to_size,
    ensure_vol_shape,
    rotate_tsdf,
    init_ray,
    get_ray_fn
)
from utils.rotation import multiply_quat, quat_to_euler, normal_to_quat, invert_quat
from utils.tsdfHelper import TSDFHelper, extend_to_bottom, get_single_biggest_cc_single
from environment.camera import SimCameraBase
from real_world.dataset import REAL_DATASET
from real_world.utils import (
    get_empty_depth,
    get_obj_bounds,
    get_kit_bounds,
    get_obj_masks_tilted,
    get_workspace_bounds,
    get_kit_crop_bounds,
    get_kit_bounds_mask,
    load_mesh_old_urdf,
    remove_pbc_ids,
    associate_pred_masks_to_gt_masks,
)
from real_world.data_label import get_crop_bounds
from real_world.gen_kit_unvisible_view_mask import get_kit_unvisible_vol_indices 
from evaluate.evaluate_model import dump_seg_vis
from data_generation import get_center_pt_from_d
from environment.utils import get_surrounding_cameras
from evaluate.html_vis import visualize_helper 
from utils.ravenutils import np_unknown_cat
from learning.srg import SRG
from learning.vol_match_rotate import VolMatchRotate
from learning.vol_match_transport import VolMatchTransport
import ray

def dump_vol(vol, voxel_size, key, dump_root, disable_cache = False, render_gif: bool = True):
    mesh_path = dump_root / f"{key}.obj"
    gif_path = dump_root / f"{key}.gif"
    generate = disable_cache or not mesh_path.exists() or not gif_path.exists()
    if generate:
        gif_path, _ = dump_vol_render_gif(
            vol,
            mesh_path,
            voxel_size,
            visualize_mesh_gif=render_gif,
            visualize_tsdf_gif=False,
            gif_num_images=10,
        )
    return mesh_path, gif_path

def get_vol(views, crop_bounds, voxel_size, key, dump_root, disable_cache=False):
    tsdf_path = dump_root / f"{key}.npy"
    if tsdf_path.exists() and not disable_cache:
        return np.load(tsdf_path)
    else:
        tsdf = TSDFHelper.tsdf_from_camera_data(views, crop_bounds, voxel_size)
        np.save(tsdf_path, tsdf)
    return tsdf

def get_vol_cameras(cameras, crop_bounds, voxel_size, key, dump_root, disable_cache=False):
    tsdf_path = dump_root / f"{key}.npy"
    if tsdf_path.exists() and not disable_cache:
        return np.load(tsdf_path)
    views = list()
    for i, camera in enumerate(cameras):
        rgb, d, _ = camera.get_image(seg_mask=False)
        views.append((rgb, d, camera.intrinsics, camera.pose_matrix))
    return get_vol(views, crop_bounds, voxel_size, key, dump_root, disable_cache)

def simulate_user_input(gt_pos, gt_ori, p0_vol, kit_vol_shape, voxel_size, max_perturb_delta, max_perturb_angle):
    # Perturb Position
    perturb_delta = np.random.randint(low=-max_perturb_delta, high=max_perturb_delta, size=3)
    gt_pos_perturbed = gt_pos - perturb_delta * voxel_size
    p1_coords = (gt_pos / voxel_size + kit_vol_shape/2).astype(int)
    p1_coords_perturbed = (gt_pos_perturbed / voxel_size + kit_vol_shape/2).astype(int)
    # Perturb orientation
    perturb_phi = max_perturb_angle * np.random.rand()
    perturb_theta = np.pi * 2 * np.random.rand()
    vec = np.array([0,-np.sin(perturb_phi),np.cos(perturb_phi)])
    rotm = np.array([[np.cos(perturb_theta), -np.sin(perturb_theta), 0],
                    [np.sin(perturb_theta), np.cos(perturb_theta), 0],
                    [0,0,1]])
    vec = rotm @ vec
    perturb_quat = normal_to_quat(vec)
    perturb_angles = quat_to_euler(perturb_quat, degrees=False)
    rotate_angles = quat_to_euler(gt_ori, degrees=False)
    p0_vol_rotate = rotate_tsdf(p0_vol, rotate_angles, degrees=False)
    p0_vol_rotate = rotate_tsdf(p0_vol_rotate, perturb_angles,  degrees=False)
    p1_ori_final = invert_quat(perturb_quat)
    gt_ori_perturbed = multiply_quat(gt_ori, perturb_quat)
    return gt_pos_perturbed, gt_ori_perturbed, p0_vol_rotate, p1_coords, p1_coords_perturbed, p1_ori_final

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    PYBULLET_GUI = False
    USE_SEG_PRED = True
    USE_SC_PRED = USE_SEG_PRED or True
    USE_KIT_SC_PRED = True
    DO_SNAP = False
    REGENERATE = False # use generated *.npy, *.obj, *.gif unless True
    SAVE_DATASET = True

    C1 = np.array([89, 130, 33 , 255 / 2]) / 255
    C2 = np.array([66, 135, 245, 255 / 2]) / 255


    # load the dataset
    dataset_root = 'dataset/eval_realworld'
    logs_dir = Path("real_world/evaluate_dataset/")
    dataset = REAL_DATASET(Path("real_world/dataset/"))
    camera_pose = dataset.camera_pose
    camera_depth_intr = dataset.camera_depth_intr

    # load bounds
    print("==> FIXME <=== Loading bounds from file. It should be properly loaded with dataset")
    bounds_ws = get_workspace_bounds()
    bounds_obj = get_obj_bounds()
    bounds_kit = get_kit_bounds()
    voxel_size = cfg.env.voxel_size
    kit_vol_shape = np.array(cfg.env.kit_vol_shape)
    kit_ws_crop_bounds = get_crop_bounds(bounds_kit, kit_vol_shape, voxel_size)
    obj_ws_crop_bounds = get_crop_bounds(
        bounds_obj, kit_vol_shape, voxel_size)  # kit_vol_shape is intentional
    # shape of input to shape completion model
    obj_vol_shape = np.array(cfg.env.obj_vol_shape)

    # Setup sim camera to mimic real world camera
    image_size = tuple(cfg.env.image_size)
    camera_pose_tmp = np.copy(camera_pose)
    camera_pose_tmp[:, 1:3] = -camera_pose_tmp[:, 1:3]
    view_matrix = np.linalg.inv(camera_pose_tmp.T).flatten()
    focal_length = camera_depth_intr[1, 1]
    sim_camera_primary = SimCameraBase(
        view_matrix=view_matrix, image_size=image_size, focal_length=focal_length)
    sim_cameras = get_surrounding_cameras(
        bounds_ws * 1.5,
        bounds_ws.mean(axis=1),
        image_size,
        less_cameras=False
    )

    seg_score_threshold = cfg.perception.seg.mask_score_threshold
    seg_threshold = cfg.perception.seg.mask_threshold
    obj_vol_shape = np.array(cfg.vol_match_6DoF.p0_vol_shape_gen)
    hw = np.ceil(obj_vol_shape / 2).astype(int) # half width
    device = get_device()
    device_cpu = torch.device("cpu")

    kit_vol_shape = np.array(cfg.vol_match_6DoF.p1_vol_shape_gen)
    kit_vol_size = kit_vol_shape * voxel_size
    kit_crop_bounds = get_kit_crop_bounds(bounds_kit, kit_vol_size)
    kit_unvisible_vol_indices = get_kit_unvisible_vol_indices()
    kit_ws_cameras = get_surrounding_cameras(
        bounds_kit,
        bounds_kit.mean(axis=1),
        image_size=image_size,
        less_cameras=False
    )

    def evaluate_sample(dataset_i):
        pbc = BulletClient(pybullet.GUI if PYBULLET_GUI else pybullet.DIRECT)
        pbc.setPhysicsEngineParameter(enableFileCaching=0)
        rgb, d, _, datapoint_path = dataset.__getitem__(
            dataset_i,
            use_idx_as_datapoint_folder_name=True,
        )
        dump_path = dict()
        dump_root = ensure_makedirs(logs_dir / f"{dataset_i}")
        dump_path["rgb"] = save_img(rgb, dump_root / "rgb.png")
        dump_path["d"] = save_img(d, dump_root / "d.png", cmap="jet")
        
        gt_labels_path = datapoint_path / "gt_labels.json"
        if not gt_labels_path.exists():
            print(f"Note: Missing gt_labels.json for {datapoint_path}")
            return dump_path
        
        # ======= segmentation mask ==============
        print('Start to evaluate segmentation.')
        with open(gt_labels_path) as gt_json_fp:
            gt_data = json.load(gt_json_fp)
        pbc_ids = list()
        for obj_data in gt_data["obj"]:
            pbc_id = load_mesh_old_urdf(obj_data, pbc)
            obj_data["pbc_id"] = pbc_id
            pbc_ids.append(pbc_id)
        for kit_data in gt_data["kit"]:
            pbc_id = load_mesh_old_urdf(kit_data, pbc)
            kit_data["pbc_id"] = pbc_id
            pbc_ids.append(pbc_id)
        _, _, raw_masks = sim_camera_primary.get_image(seg_mask=True)
        gt_masks = None
        for obj_id, obj_data in enumerate(gt_data["obj"]):
            mask = (raw_masks == obj_data["pbc_id"]).astype(int)
            obj_data["gt_mask"] = mask
            gt_masks = np_unknown_cat(gt_masks, mask)
            obj_data["obj_id"] = obj_id

        srg = None
        if USE_SEG_PRED or USE_SC_PRED:
            srg = SRG(cfg.perception, hw, device_cpu)

        matched_pred_masks = [np.zeros_like(gt_masks[0]) for _ in range(len(gt_masks))]
        if USE_SEG_PRED:
            pred_masks = get_obj_masks_tilted(rgb, d, camera_pose, camera_depth_intr, cfg, srg)
            if pred_masks is None:
                print("No object masks predicted by segmentation network.")
            else:
                gt_mask_to_pred_mask_matching = associate_pred_masks_to_gt_masks(
                    pred_masks,
                    gt_masks,
                )
                for gt_mask_index, (match_index, _) in enumerate(gt_mask_to_pred_mask_matching):
                    if match_index is not None:
                        matched_pred_masks[gt_mask_index] = pred_masks[match_index]
                    gt_data["obj"][gt_mask_index]["pred_mask"] = matched_pred_masks[gt_mask_index] 
            if SAVE_DATASET:
                for obj_id, obj_data in enumerate(gt_data["obj"]):
                    output_dir = Path(f'{dataset_root}/seg/{dataset_i}_{obj_id}')
                    output_dir.mkdir(exist_ok=True, parents=True)
                    with h5py.File(str(output_dir / "data.hdf"), "w") as hdf:
                        hdf.create_dataset("gt_mask", data=obj_data['gt_mask'])
                        hdf.create_dataset("pred_mask", data=obj_data['pred_mask'])
        dump_path.update(
            dump_seg_vis(
                img=rgb,
                boxes=list(),
                target_boxes=list(),
                scores=np.ones(gt_masks.shape[0]),
                masks=matched_pred_masks,
                target_masks=gt_masks,
                score_threshold=seg_score_threshold,
                mask_threshold=seg_threshold,
                log_path=dump_root,
            )
        )
        pbc_ids = remove_pbc_ids(pbc, pbc_ids)
        sc_mask_key = "pred_mask" if USE_SEG_PRED else "gt_mask"

        # ============= obj shape completion ==============
        # using gt mask, figure out the center of crop
        print("Start to evaluate obj sc.")
        for obj_data in gt_data["obj"]:
            obj_id = obj_data["obj_id"]
            mask = obj_data[sc_mask_key]
            masked_d = get_masked_d(mask, d)
            center_pt = get_center_pt_from_d(
                masked_d, camera_depth_intr, camera_pose, bounds_ws)
            if center_pt is None:
                print("Center pt none")
                continue
            crop_bounds = get_bounds_from_center_pt(
                center_pt, obj_vol_shape, voxel_size, bounds_obj)
            views = [(rgb, masked_d, camera_depth_intr, camera_pose)]
            sc_inp_key = f"{obj_id}_sc_inp"
            sc_inp = get_vol(
                views, crop_bounds, voxel_size, sc_inp_key, dump_root, disable_cache=REGENERATE)
            obj_data["raw_vol"] = pad_crop_to_size(sc_inp, obj_vol_shape)
            obj_data["raw_vol_mesh_path"], dump_path[sc_inp_key] = dump_vol(
                obj_data["raw_vol"], voxel_size, sc_inp_key, dump_root, disable_cache=REGENERATE)
            obj_data["raw_vol_origin"] = crop_bounds.mean(axis=1)

            # save gt as well:
            sc_targ_key = f"{obj_id}_sc_target"
            pbc_id = load_mesh_old_urdf(obj_data, pbc)
            sc_target = get_vol_cameras(
                sim_cameras, crop_bounds, voxel_size, sc_targ_key, dump_root, disable_cache=REGENERATE)
            pbc.removeBody(pbc_id)
            obj_data["gt_vol_mesh_path"], dump_path[sc_targ_key]= dump_vol(
                sc_target, voxel_size, sc_targ_key, dump_root, disable_cache=REGENERATE)
            obj_data["gt_vol"] = sc_target

            obj_data["snap_obj_vol"] = obj_data["gt_vol"]
            obj_data["snap_obj_mesh"] = obj_data["gt_vol_mesh_path"]

            if USE_SC_PRED:
                sc_pred_key = f"{obj_id}_sc_pred"
                sc_pred_tsdf_path = dump_root / f'{sc_pred_key}.npy'
                if sc_pred_tsdf_path.exists() and not REGENERATE:
                    sc_pred = np.load(sc_pred_tsdf_path)
                else:
                    sc_inp = torch.tensor(obj_data["raw_vol"], device=srg.device)
                    sc_inp = sc_inp.unsqueeze(dim=0).unsqueeze(dim=0)
                    sc_pred = srg.sc_model(sc_inp).detach().squeeze().cpu().numpy()
                    sc_pred = get_single_biggest_cc_single(sc_pred)
                    sc_pred = extend_to_bottom(sc_pred)
                    np.save(sc_pred_tsdf_path, sc_pred)
                obj_data["pred_vol"] = sc_pred
                obj_data["pred_vol_mesh_path"], dump_path[sc_pred_key] = dump_vol(
                    sc_pred, voxel_size, sc_pred_key, dump_root, disable_cache=REGENERATE)
                obj_data["snap_obj_vol"] = obj_data["pred_vol"]
                obj_data["snap_obj_mesh"] = obj_data["pred_vol_mesh_path"]

                if SAVE_DATASET:
                    output_dir = Path(f'{dataset_root}/sc_obj/{dataset_i}_{obj_id}')
                    output_dir.mkdir(exist_ok=True, parents=True)
                    with h5py.File(str(output_dir / "data.hdf"), "w") as hdf:
                        hdf.create_dataset("gt_vol", data=sc_target)
                        hdf.create_dataset("pred_vol", data=sc_pred)
                        hdf.create_dataset("raw_vol", data=sc_inp)
                
        if USE_SEG_PRED:  # free up memory for snap-net models
            del srg
            torch.cuda.empty_cache()

        # ========================= kit shape completion ==========================
        print('Start to evaluate kit sc.')
        kit_mask = get_kit_bounds_mask(camera_pose, camera_depth_intr, rgb.shape[:2])
        kit_depth = get_masked_d(kit_mask, d)
        views = [(rgb, kit_depth, camera_depth_intr, camera_pose)]
        kit_sc_inp_key = 'kit_sc_inp'
        kit_sc_inp = get_vol(
            views, kit_crop_bounds, voxel_size, kit_sc_inp_key, dump_root, disable_cache=REGENERATE)
        kit_sc_inp = ensure_vol_shape(kit_sc_inp, kit_vol_shape)
        kit_sc_inp[kit_unvisible_vol_indices] = 1  # Commented because this is buggy right now
        dump_vol(kit_sc_inp, voxel_size, kit_sc_inp_key, dump_root, disable_cache=REGENERATE)
        
        # save gt as well:
        kit_sc_target_key = 'kit_sc_target'
        pbc_ids = list()
        for kit_data in gt_data["kit"]:
            pbc_id = load_mesh_old_urdf(kit_data, pbc)
            pbc_ids.append(pbc_id)
        kit_sc_target = get_vol_cameras(
            sim_cameras, kit_crop_bounds, voxel_size, kit_sc_target_key, dump_root, disable_cache=REGENERATE)
        kit_sc_target = ensure_vol_shape(kit_sc_target, kit_vol_shape)
        remove_pbc_ids(pbc, pbc_ids)
        pbc_ids = list()
        kit_sc_target_mesh_path, _ = dump_vol(
            kit_sc_target, voxel_size, kit_sc_target_key, dump_root, disable_cache=REGENERATE)

        snap_kit_vol = kit_sc_target
        snap_kit_mesh = kit_sc_target_mesh_path
        if USE_KIT_SC_PRED:
            kit_sc_pred_key = 'kit_sc_pred'
            kit_sc_pred_tsdf_path = dump_root / f'{kit_sc_pred_key}.npy'
            sc_kit_model = None
            #print("=======>FIXME<======= using device_cpu for kit shape completion")
            if kit_sc_pred_tsdf_path.exists() and not REGENERATE:
                kit_sc_pred = np.load(kit_sc_pred_tsdf_path)
            else:
                sc_kit_model = torch.load(Path(cfg.perception.sc_kit.path), map_location=device_cpu)
                kit_sc_inp_ten = torch.tensor(kit_sc_inp, device=device_cpu)
                kit_sc_inp_ten = kit_sc_inp_ten.unsqueeze(dim=0).unsqueeze(dim=0)
                kit_sc_pred = sc_kit_model(kit_sc_inp_ten)
                kit_sc_pred = kit_sc_pred.squeeze().detach().cpu().numpy()
                kit_sc_pred = get_single_biggest_cc_single(kit_sc_pred)
                np.save(kit_sc_pred_tsdf_path, kit_sc_pred)
            kit_sc_pred_mesh_path, _ = dump_vol(
                kit_sc_pred, voxel_size, kit_sc_pred_key, dump_root, disable_cache=REGENERATE)
            del sc_kit_model
            torch.cuda.empty_cache()
            snap_kit_vol = kit_sc_pred
            snap_kit_mesh = kit_sc_pred_mesh_path
            if SAVE_DATASET:
                output_dir = Path(f'{dataset_root}/sc_kit/{dataset_i}_{obj_id}')
                output_dir.mkdir(exist_ok=True, parents=True)
                with h5py.File(str(output_dir / "data.hdf"), "w") as hdf:
                    hdf.create_dataset("gt_vol", data=kit_sc_target)
                    hdf.create_dataset("pred_vol", data=kit_sc_pred)
                    hdf.create_dataset("raw_vol", data=kit_sc_inp)

        # ========================= Save Dataset ==========================
        print("Start to evaluate snapping")
        # For figuring out the (kit,obj) pairs, let's load the gt_ids.json
        with open(datapoint_path / "gt_ids.json", "r") as gt_ids_fp:
            gt_ids = json.load(gt_ids_fp)

        all_pos, all_pred_masks = [], []
        all_p1_coords, all_p1_ori = [], []
        all_p0_vol_raw, all_p0_vol_sc, all_p0_vol = [], [], []

        for _, kit_obj_pair in enumerate(gt_ids):
            if "kit" not in kit_obj_pair or "obj" not in kit_obj_pair:
                # This object does not have a corresponding kit or vice versa
                continue
            kit_mesh_path = kit_obj_pair["kit"]
            obj_mesh_path = kit_obj_pair["obj"]
            def find_data(mesh_path, meshes_data):
                for mesh_data in meshes_data:
                    if mesh_data["mesh_path"] == mesh_path:
                        return mesh_data
                return None
            kit_data = find_data(kit_mesh_path, gt_data["kit"])
            obj_data = find_data(obj_mesh_path, gt_data["obj"])
            if obj_data is None or kit_data is None:
                print(f"ERROR: obj/kit data not found for {obj_mesh_path}-{kit_mesh_path}.\
                    FIX gt_labels.json")

            gt_obj_pose__world_frame = (obj_data["gt_pos"], obj_data["gt_ori"])
            gt_obj_pose__raw_vol_frame = (
                gt_obj_pose__world_frame[0] - obj_data["raw_vol_origin"],
                gt_obj_pose__world_frame[1]
            )
            raw_vol_pose__gt_obj_frame = pbc.invertTransform(*gt_obj_pose__raw_vol_frame)
            aligned_obj_pose__world_frame = (kit_data["gt_pos"], kit_data["gt_ori"])
            aligned_raw_vol_pose__world_frame = pbc.multiplyTransforms(
                *aligned_obj_pose__world_frame,
                *raw_vol_pose__gt_obj_frame
            )
            obj_data['gt_pos_kit_frame'] = aligned_raw_vol_pose__world_frame[0]-kit_crop_bounds.mean(axis=1)
            obj_data['gt_ori_kit_frame'] = aligned_raw_vol_pose__world_frame[1]

            p1_coords = (obj_data['gt_pos_kit_frame'] / voxel_size + kit_vol_shape/2).astype(int)
            p1_ori = obj_data['gt_ori_kit_frame']
            all_p0_vol_raw.append(obj_data['raw_vol'])
            all_p0_vol_sc.append(obj_data['pred_vol'])
            all_p0_vol.append(obj_data['gt_vol'])
            all_pos.append(obj_data["gt_pos"])
            all_pred_masks.append(obj_data["pred_mask"])
            all_p1_coords.append(p1_coords)
            all_p1_ori.append(p1_ori)


        if SAVE_DATASET:
            output_dir = Path(f'{dataset_root}/snap/{dataset_i}')
            output_dir.mkdir(exist_ok=True, parents=True)
            kit_mask = np.ones(rgb.shape[:2])
            d_empty = get_empty_depth()
            kit_mask[abs(d-d_empty)<0.01] = 0
            kit_mask[:, :640] = 0
            kit_mask[:200, :] = 0
            kit_mask[-50:, :] = 0
            with h5py.File(str(output_dir / "data.hdf"), "w") as hdf:
                hdf.create_dataset("rgb", data=rgb)
                hdf.create_dataset("d", data=d)
                hdf.create_dataset("all_pos", data=np.array(all_pos))
                hdf.create_dataset("all_pred_masks", data=np.array(all_pred_masks))
                hdf.create_dataset("kit_pos", data=np.mean(kit_crop_bounds, axis=1))
                hdf.create_dataset("p1_vol_raw", data=kit_sc_inp)
                hdf.create_dataset("p1_vol_sc", data=kit_sc_pred)
                hdf.create_dataset("p1_vol", data=kit_sc_target)
                hdf.create_dataset("p1_mask", data=kit_mask)
                hdf.create_dataset("all_p0_vol_raw", data=np.array(all_p0_vol_raw))
                hdf.create_dataset("all_p0_vol_sc", data=np.array(all_p0_vol_sc))
                hdf.create_dataset("all_p0_vol", data=np.array(all_p0_vol))
                hdf.create_dataset("all_p1_coords", data=np.array(all_p1_coords))
                hdf.create_dataset("all_p1_ori", data=np.array(all_p1_ori))

        # ========================= Snap ==========================

        if DO_SNAP:
            vis_env = MeshRendererEnv(gui=False)
            # generate gt alignment
            vis_env.load_mesh(snap_kit_mesh, rgba=C1)
            for obj_data in gt_data["obj"]:
                vis_env.load_mesh(Path(obj_data['snap_obj_mesh']), obj_data['gt_pos_kit_frame'], obj_data['gt_ori_kit_frame'], rgba=C2)
            dump_path['gt_alignment'] = vis_env.render(dump_root / f"gt_alignment.gif")
            vis_env.reset()

            vm_cfg = cfg.vol_match_6DoF 
            max_perturb_delta = np.array(vm_cfg.max_perturb_delta)
            max_perturb_angle = float(vm_cfg.max_perturb_angle)
            transporter = VolMatchTransport.from_cfg(vm_cfg, cfg.env.voxel_size, load_model=True, log=False)
            rotator = VolMatchRotate.from_cfg(vm_cfg, load_model=True, log=False)
            for obj_id, obj_data in enumerate(gt_data["obj"]):
                gt_pos = obj_data['gt_pos_kit_frame']
                gt_ori = obj_data['gt_ori_kit_frame']
                p0_vol = obj_data['snap_obj_vol']
                p1_vol = snap_kit_vol
                # <=== simulate user input ===>
                gt_pos_perturbed, gt_ori_perturbed, p0_vol_rotate, p1_coords, p1_coords_perturbed, p1_ori_final = \
                    simulate_user_input(gt_pos, gt_ori, p0_vol, kit_vol_shape, voxel_size, max_perturb_delta, max_perturb_angle)
                # save simulated user input for visualization
                obj_data['gt_pos_perturbed'] = gt_pos_perturbed
                obj_data['gt_ori_perturbed'] = gt_ori_perturbed
                # prepare data sample
                p0_vol_rotate_ten = torch.tensor(p0_vol_rotate, device=device).unsqueeze(dim=0)
                p1_vol_ten = torch.tensor(p1_vol, device=device).unsqueeze(dim=0)
                user_coords_gt_ten = torch.tensor(p1_coords, device=device).unsqueeze(dim=0)
                user_coords_ten = torch.tensor(p1_coords_perturbed, device=device).unsqueeze(dim=0)
                p1_ori_gt_ten = torch.tensor(p1_ori_final, device=device).unsqueeze(dim=0)
                batch = {
                    "p0_vol": p0_vol_rotate_ten,
                    "p1_vol": p1_vol_ten, 
                    "p1_coords": user_coords_gt_ten,
                    "p1_coords_user": user_coords_ten,
                    "p1_ori": p1_ori_gt_ten,
                    "concav_ori": torch.tensor([[0, 0, 0, 1]], device=device),
                    "symmetry": torch.tensor([[-1, -1, -1]]),
                }
                with torch.no_grad():
                    _, pred_coords, _, pos_diff = transporter.run(batch, training=False, log=False, calc_loss=True)
                    batch['p1_coords'] = pred_coords.astype(int)
                    _, _, pred_ori, rot_diff = rotator.run(batch, training=False, log=False, calc_loss=True)
                    # print(pos_diff, rot_diff)
                    pred_coords = pred_coords[0]
                # save predicted user input for visualization
                obj_data['pred_pos'] = (pred_coords-kit_vol_shape/2) * voxel_size
                obj_data['pred_ori'] = multiply_quat(gt_ori_perturbed, pred_ori)
            del transporter
            del rotator

            # Load meshes at simulated user input
            vis_env.load_mesh(snap_kit_mesh, rgba=C1)
            for obj_data in gt_data["obj"]:
                vis_env.load_mesh(Path(obj_data['snap_obj_mesh']), obj_data['gt_pos_perturbed'], obj_data['gt_ori_perturbed'], rgba=C2)
            dump_path['sim_user_alignment'] = vis_env.render(dump_root / f"sim_user_alignment.gif")
            vis_env.reset()

            # Load meshes at predicted pose
            vis_env.load_mesh(snap_kit_mesh, rgba=C1)
            for obj_data in gt_data["obj"]:
                vis_env.load_mesh(Path(obj_data['snap_obj_mesh']), obj_data['pred_pos'], obj_data['pred_ori'], rgba=C2)
            dump_path['pred_alignment'] = vis_env.render(dump_root / f"pred_alignment.gif")
            vis_env.reset()

        return dump_path

    use_ray = init_ray(cfg.ray)
    fn = get_ray_fn(cfg.ray, evaluate_sample, gpu_frac=0.05)
    tasks = list()        
    for dataset_i in range(len(dataset)):
        tasks.append(fn(dataset_i))
    if use_ray:
        tasks = ray.get(tasks)
    
    if len(tasks) > 0:
        cols = list(tasks[0].keys())
        title = f"SEG_PRED-{USE_SEG_PRED}--SC_PRED-{USE_SC_PRED}--KIT_SC_PRED-{USE_KIT_SC_PRED}"
        visualize_helper(tasks, logs_dir, cols, title=title)


if __name__ == "__main__":
    main()
