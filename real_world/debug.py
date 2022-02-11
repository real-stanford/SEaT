# This file contains code for dumping the volumes for both the object and the kit!!
import sys
from pathlib import Path
import time
root_path = Path(__file__).parent.parent.absolute()
# print("Adding to python path: ", root_path)
sys.path = [str(root_path)] + sys.path

import numpy as np
from environment.real.cameras import RealSense
from matplotlib import pyplot as plt
from utils.tsdfHelper import TSDFHelper
from utils.rotation import multiply_quat
from random import sample
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import hydra
from omegaconf import DictConfig
from data_generation import get_center_pt_from_d
import json
from environment.real.ur5 import UR5_URX
from learning.srg import SRG
import pybullet as p
from real_world.rw_utils import color_mask_rgb, ensure_minus_pi_to_pi, fix_ur5_rotation, get_crops_wb, get_kit_bounds, get_kit_bounds_mask, get_obj_bounds, get_obj_masks_tilted, get_workspace_bounds, \
    get_tool_init, get_obj_masks, get_client_frame_pose, transform_mask, get_obj_masks_tilted, clip_angle
from environment.utils import SCENETYPE
from PIL import Image
from utils.pointcloud import PointCloud
from utils.tsdfHelper import get_single_biggest_cc_single, extend_to_bottom
import open3d as o3d
import torch
from utils import ensure_vol_shape, get_device, get_masked_rgb, get_masked_d, \
    get_bounds_from_center_pt, mkdir_fresh, pad_crop_to_size, rotate_tsdf, show_overlay_image, center_crop
from environment.meshRendererEnv import dump_vol_render_gif, dump_tsdf_vis, MeshRendererEnv
from shutil import rmtree
from evaluate.evaluate_model import dump_seg_vis
import math3d as m3d
from icecream import ic as print_ic
from learning.vol_match_rotate import VolMatchRotate
from learning.vol_match_transport import VolMatchTransport
from real_world.gen_kit_unvisible_view_mask import get_kit_unvisible_vol_indices 
from scipy.signal import convolve2d
from real_world.dataset import REAL_DATASET
import pickle
from evaluate.html_vis import visualize_helper

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    
    DEBUG_SNAP = True

    device = get_device()
    device_cpu = torch.device("cpu")
    vm_cfg = cfg.vol_match_6DoF 
    p1_vol_shape = np.array(vm_cfg.p1_vol_shape)
    bounds_kit = get_kit_bounds()
    voxel_size = cfg.env.voxel_size
    
    debug_i = 3
    debug_root = Path(f"real_world/debug")
    debug_path = debug_root / f'{debug_i}'

    device = get_device()
    sc_obj_model_path = 'logs/sc_object/sc_50.pth'
    sc_obj_model = torch.load(sc_obj_model_path, map_location=device_cpu).eval()

    for i in range(4):
        p0_vol = np.load(debug_path / f'{i}_inp.npy')
        p0_vol_ten = torch.tensor(p0_vol).to(device).float()
        p0_vol_ten = p0_vol_ten.unsqueeze(0).unsqueeze(0)

        p0_vol_sc = sc_obj_model(p0_vol_ten).squeeze().cpu().detach().numpy()
        p0_vol_sc = get_single_biggest_cc_single(p0_vol_sc)
        p0_vol_sc = extend_to_bottom(p0_vol_sc)

        dump_vol_render_gif(p0_vol_sc, debug_path / f"{i}_out.obj", voxel_size, visualize_mesh_gif=False, visualize_tsdf_gif=False)

    # transporter = VolMatchTransport.from_cfg(vm_cfg, cfg.env.voxel_size, load_model=True, log=False)
    # rotator = VolMatchRotate.from_cfg(vm_cfg, load_model=True, log=False)

    # diff_dict = pickle.load(open(debug_path / 'diff_dict.pkl', 'rb'))
    # name_vols = pickle.load(open(debug_path /'name_vols.pkl', 'rb'))
    # name_transformations = pickle.load(open(debug_path /'name_transformations.pkl', 'rb'))

    # tasks = list()

    # for name, val in diff_dict.items():
    #     p0_vol = name_vols[name]    
    #     p1_vol = name_vols['kit']
    #     print_ic(p0_vol.shape, p1_vol.shape)
    #     world__p1_pos, _ = p.multiplyTransforms(val["upd_pos"], val["upd_ori"], -name_transformations[name], p.getQuaternionFromEuler([0, 0, 0]))
    #     # world__p1_pos = np.array(val["upd_pos"])
    #     kit__p1_pos = world__p1_pos - bounds_kit[:, 0]
    #     user_coords = (kit__p1_pos / voxel_size).astype(int)
    #     user_ori = val["upd_ori"]

    #     # Rotate the p0_vol according to the user provided input
    #     rotate_angles = np.array(p.getEulerFromQuaternion(user_ori))
    #     p0_vol_rotate = rotate_tsdf(p0_vol, rotate_angles)
    #     p0_vol_rotate_ten = torch.tensor(p0_vol_rotate, device=device).unsqueeze(dim=0)
    #     p1_vol_ten = torch.tensor(p1_vol, device=device).unsqueeze(dim=0)
    #     print_ic(user_coords, user_ori)
    #     user_coords_ten = torch.tensor(user_coords, device=device).unsqueeze(dim=0)
    #     batch = {
    #         "p0_vol": p0_vol_rotate_ten,
    #         "p1_vol": p1_vol_ten, 
    #         "p1_coords": None,
    #         "p1_coords_user": user_coords_ten,
    #         "p1_ori": None,
    #         "concav_ori": torch.tensor([[0, 0, 0, 1]], device=device),
    #         "symmetry": torch.tensor([[-1, -1, -1]]),
    #     }
    #     pred_coords, pred_ori = None, None
    #     with torch.no_grad():
    #         since = time.time()
    #         _, pred_coords, _, _ = transporter.run(
    #             batch, training=False, log=False, calc_loss=False)
    #         print(f"Position prediction finished in {time.time() - since}")
    #         since = time.time()
    #         batch['p1_coords'] = pred_coords.astype(int)
    #         _, _, pred_ori, _ = rotator.run(
    #             batch, training=False, log=False, calc_loss=False)
    #         print(f"Rotation prediction finished in {time.time() - since}")
    #         pred_coords = pred_coords[0]
    #         print_ic(pred_coords, pred_ori)


    #     if DEBUG_SNAP:
    #         p1_vol_crop = center_crop(p1_vol, user_coords, p1_vol_shape, tensor=False)
    #         pred_pos = (pred_coords - user_coords) * voxel_size
    #         pred_ori_given_user = multiply_quat(user_ori, pred_ori)

    #         p0_mesh_path = debug_path / f"{name}_p0_vol.obj"
    #         if not p0_mesh_path.exists():
    #             TSDFHelper.to_mesh(p0_vol, p0_mesh_path, voxel_size)
    #         p1_mesh_path = debug_path / f"{name}_p1_vol.obj"
    #         if not p1_mesh_path.exists():
    #             TSDFHelper.to_mesh(p1_vol_crop, p1_mesh_path, voxel_size)

    #         bb_min = -0.08 * np.ones((3,))
    #         bb_max = 0.08 * np.ones((3,))

    #         vis_env = MeshRendererEnv(gui=False)
    #         vis_env.load_mesh(p1_mesh_path, rgba=np.array([1, 0, 0, 0.5]))
    #         vis_env.load_mesh(p0_mesh_path, [0,0,0], user_ori, rgba=np.array([0, 1, 0, 0.5]))
    #         vis_env.render(debug_path / f"{name}_user_vis.gif", bb_min, bb_max)
            
    #         vis_env.reset()
    #         vis_env.load_mesh(p1_mesh_path, rgba=np.array([1, 0, 0, 0.5]))
    #         vis_env.load_mesh(p0_mesh_path, pred_pos, pred_ori_given_user, rgba=np.array([0, 1, 0, 0.5]))
    #         vis_env.render(debug_path / f"{name}_pred_vis.gif", bb_min, bb_max)

    #     tasks.append({'user': f"{name}_user_vis.gif", 'pred': f"{name}_pred_vis.gif"})

    # cols = ["user", "pred"]
    # html_file_name = "debug_snap.html"
    # visualize_helper(tasks, debug_path, cols, html_file_name=html_file_name)


if __name__ == "__main__":
    main()

    