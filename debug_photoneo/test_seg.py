
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
from utils.tsdfHelper import TSDFHelper, extend_to_bottom
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
    get_tool_init, get_obj_masks, get_client_frame_pose, transform_mask, get_obj_masks_tilted, clip_angle, get_kit_crop_bounds
from environment.utils import SCENETYPE, get_tableau_palette
from PIL import Image
from utils.pointcloud import PointCloud
from utils.tsdfHelper import get_single_biggest_cc_single
import open3d as o3d
import torch
from utils import ensure_vol_shape, get_device, get_masked_rgb, get_masked_d, \
    get_bounds_from_center_pt, mkdir_fresh, pad_crop_to_size, rotate_tsdf, show_overlay_image, center_crop
from environment.meshRendererEnv import dump_vol_render_gif, dump_tsdf_vis, MeshRendererEnv
from shutil import rmtree
from evaluate.evaluate_model import dump_seg_vis
import math3d as m3d
from icecream import ic as print_ic
# from learning.vol_match_rotate import VolMatchRotate
# from learning.vol_match_transport import VolMatchTransport
# from real_world.gen_kit_unvisible_view_mask import get_kit_unvisible_vol_indices 
from real_world.dataset import REAL_DATASET
import cv2

@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    device = torch.device('cpu')

    dataset = REAL_DATASET(Path("real_world/dataset/"))
    camera_pose = dataset.camera_pose
    camera_color_intr = dataset.camera_depth_intr
    voxel_size = cfg.env.voxel_size
    kit_vol_shape = np.array(cfg.env.kit_vol_shape)
    kit_vol_size = kit_vol_shape * voxel_size
    bounds_ws = get_workspace_bounds()
    bounds_obj = get_obj_bounds()
    bounds_kit = get_kit_bounds()
    kit_crop_bounds = get_kit_crop_bounds(bounds_kit, kit_vol_size)

    obj_vol_shape = np.array(cfg.env.obj_vol_shape)
    hw = np.ceil(obj_vol_shape / 2).astype(int) # half width
    srg = SRG(cfg.perception, hw, device)  
    sc_kit_model = torch.load(Path(cfg.perception.sc_kit.path), map_location=device)

    n = len(dataset)
    debug_root = Path(f"debug_photoneo/debug_sc4")
    # print("Please enter the datapoint number [from 0-23]:")
    # debug_dir = int(input())
    for i in range(3):
        debug_path = debug_root / f'{i}'
        debug_path.mkdir(exist_ok=True, parents=True)
        rgb, d, _, _ = dataset.__getitem__(i, use_idx_as_datapoint_folder_name=True)
        plt.imshow(rgb)
        plt.savefig(debug_path / "rgb.png")
        plt.close()
        plt.imshow(d)
        plt.savefig(debug_path / "d.png")
        plt.close()
        unfiltered_masks = get_obj_masks_tilted(rgb, d, camera_pose, camera_color_intr, cfg, srg)
        if unfiltered_masks is None:
            print("No object masks found.")
            exit(0)
        seg_area_threshold = 1000
        masks = []
        for mask in unfiltered_masks:
            if np.sum(mask==1) > seg_area_threshold:
                masks.append(mask)
        masks = np.array(masks)
        dump_seg_vis(rgb, [], [], np.ones(masks.shape[0]), masks, [], srg.seg_score_threshold, srg.seg_threshold, debug_path)

        for j, mask in enumerate(masks):
            # if np.sum(mask==1) < 3000:
            #     continue
            masked_d = get_masked_d(mask, d)
            center_pt = get_center_pt_from_d(masked_d, camera_color_intr, camera_pose, bounds_ws)
            if center_pt is None:
                print("Center pt none")
                continue
            crop_bounds = get_bounds_from_center_pt(
                center_pt, obj_vol_shape, voxel_size, bounds_obj)
            views = [(rgb, masked_d, camera_color_intr, camera_pose)]
            sc_inp = TSDFHelper.tsdf_from_camera_data(views, crop_bounds, voxel_size)
            sc_inp = ensure_vol_shape(sc_inp, obj_vol_shape)

            dump_vol_render_gif(sc_inp, debug_path / f"{j}_inp.obj", voxel_size, visualize_mesh_gif=False, visualize_tsdf_gif=False)
            sc_inp = torch.tensor(sc_inp, device=srg.device).unsqueeze(dim=0).unsqueeze(dim=0)
            obj_vol = srg.sc_model(sc_inp).detach().squeeze().cpu().numpy()
            obj_vol = get_single_biggest_cc_single(obj_vol)
            obj_vol = extend_to_bottom(obj_vol)
            dump_vol_render_gif(obj_vol, debug_path / f"{j}_out.obj", voxel_size, visualize_mesh_gif=False, visualize_tsdf_gif=False)

        # kit_mask = get_kit_bounds_mask(camera_pose, camera_color_intr, rgb.shape[:2])
        # # show_overlay_image(kit_mask, rgb)
        # kit_depth = get_masked_d(kit_mask, d)
        # views = [(rgb, kit_depth, camera_color_intr, camera_pose)]
        # #print("=======>FIXME<======= using larger truncation margin factor for kit volume")
        # kit_sc_inp = TSDFHelper.tsdf_from_camera_data(
        #     views, kit_crop_bounds, voxel_size)
        # kit_sc_inp = ensure_vol_shape(kit_sc_inp, kit_vol_shape)
        # kit_sc_inp[dataset.kit_bounds_mask] = 1

        # dump_vol_render_gif(kit_sc_inp, debug_path / f"kit_inp.obj",
        #                     voxel_size, visualize_mesh_gif=False,
        #                     visualize_tsdf_gif=False)


        # kit_sc_inp = torch.tensor(
        #     kit_sc_inp, device=device).unsqueeze(dim=0).unsqueeze(dim=0)
        # kit_vol = sc_kit_model(kit_sc_inp)
        # kit_vol = kit_vol.squeeze().detach().cpu().numpy()
        # kit_vol = get_single_biggest_cc_single(kit_vol)
        # dump_vol_render_gif(kit_vol, debug_path / f"kit_out.obj",
        #                     voxel_size, visualize_mesh_gif=False,
        #                     visualize_tsdf_gif=False)

if __name__ == "__main__":
    main()