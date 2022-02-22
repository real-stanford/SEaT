# This file contains code for dumping the volumes for both the object and the kit!!
import sys
from pathlib import Path
root_path = Path(__file__).parent.parent.absolute()
# print("Adding to python path: ", root_path)
sys.path = [str(root_path)] + sys.path

import numpy as np
import hydra
from omegaconf import DictConfig
from real_world.rw_utils import get_kit_bounds, get_kit_bounds_mask, get_kit_crop_bounds, get_obj_bounds_mask
from utils import ensure_vol_shape, get_masked_d
from real_world.dataset import REAL_DATASET
from real_world.pyphoxi import PhoXiSensor
import cv2
import os
from utils.tsdfHelper import TSDFHelper

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    os.remove('real_world/empty_depth.npy')
    os.remove('real_world/kit_bounds_mask.npy')
    os.remove('real_world/kit_mask.npy')
    os.remove('real_world/obj_mask.npy')
    bounds_kit = get_kit_bounds()
    # Setup segmentation and shape completion
    voxel_size = cfg.env.voxel_size
    # Setup crop bounds for kit sc input
    kit_vol_shape = np.array(cfg.env.kit_vol_shape)
    kit_vol_size = kit_vol_shape * voxel_size
    kit_crop_bounds = get_kit_crop_bounds(bounds_kit, kit_vol_size)
    tcp_ip = "127.0.0.1"
    tcp_port = 50200
    bin_cam = PhoXiSensor(tcp_ip, tcp_port)
    bin_cam.start()
    camera_pose = bin_cam._extr
    camera_color_intr = bin_cam._intr
    _, gray, d = bin_cam.get_frame(True)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    np.save("real_world/empty_depth.npy", d)
    obj_mask = get_obj_bounds_mask(camera_pose, camera_color_intr, d.shape)
    kit_mask = get_kit_bounds_mask(camera_pose, camera_color_intr, rgb.shape[:2])
    kit_depth = get_masked_d(kit_mask, d)
    views = [(rgb, kit_depth, camera_color_intr, camera_pose)]
    kit_sc_inp = TSDFHelper.tsdf_from_camera_data(
        views, kit_crop_bounds, voxel_size)
    kit_sc_inp = ensure_vol_shape(kit_sc_inp, kit_vol_shape)
    np.save('real_world/kit_bounds_mask.npy', kit_sc_inp)

if __name__ == "__main__":
    main()
