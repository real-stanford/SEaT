"""
For generating input to the kit shape completion network, we initialize the tsdf volume with -1
This basically means, that the tsdf generation algorithm will assume anything that is not visible
is a solid volume.

Kit view bounds are not entirely visible in the camera (because it's close to the workspace). Thus,
the tsdf generation algorithm create a solid (prismatic wedges like) volume.

We handle this issue by generating and storing these volumes for an empty kit workspace. Later, 
during test time, we mask the kit shape completion input using this volume.
"""

import sys
from pathlib import Path
root_path = Path(__file__).parent.parent.absolute()
# print("Adding to python path: ", root_path)
sys.path = [str(root_path)] + sys.path

from utils.tsdfHelper import TSDFHelper
from environment.real.cameras import RealSense
from icecream import ic as print_ic
import numpy as np
from real_world.rw_utils import get_kit_bounds
from omegaconf import DictConfig
import hydra
from utils import ensure_vol_shape


def __get_kit_unvisible_view_mask_path():
    return Path("real_world/dataset/kit_bounds_mask.npy")


def get_kit_unvisible_vol_indices():
    kit_unvisible_view_mask_path = __get_kit_unvisible_view_mask_path()
    if not kit_unvisible_view_mask_path.exists():
        print("Kit unvisible view mask cache not found. \
            Please run real_world/gen_kit_unvisible_view_mask.py file and try again")
        raise FileNotFoundError
    vol = np.load(kit_unvisible_view_mask_path)
    return (vol <= 0)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    DEBUG = True
    input("Make sure that the kit workspace is empty and press enter ...")

    # Kit view bounds:
    bounds_kit = get_kit_bounds()
    voxel_size = cfg.env.voxel_size
    p1_vol_shape = np.array(cfg.env.kit_vol_shape)
    print("USING P1 vol shape from P1 vol shape gen")
    p1_vol_size = np.array(p1_vol_shape) * voxel_size
    kit_crop_bounds = np.empty((3, 2))
    bounds_kit_center = bounds_kit.mean(axis=1)
    kit_crop_center = np.empty(3)
    kit_crop_center[:2] = bounds_kit_center[:2]
    kit_crop_center[2] = bounds_kit[2, 0] + p1_vol_size[2] / 2
    kit_crop_bounds[:, 0] = kit_crop_center - p1_vol_size / 2
    kit_crop_bounds[:, 1] = kit_crop_center + p1_vol_size / 2
    print_ic(bounds_kit, kit_crop_bounds)

    # Now, just generate the volume
    bin_cam = RealSense()
    rgb, d = bin_cam.get_camera_data(avg_depth=True)
    kit_unvisible_view_mask_path = __get_kit_unvisible_view_mask_path()
    if not kit_unvisible_view_mask_path.parent.exists():
        kit_unvisible_view_mask_path.parent.mkdir()
    np.save(kit_unvisible_view_mask_path.parent /
            f"{kit_unvisible_view_mask_path.stem}_d.npy", d)
    # generate the mask
    views = [(rgb, d, bin_cam.color_intr, bin_cam.pose)]
    vol = TSDFHelper.tsdf_from_camera_data(views, kit_crop_bounds, voxel_size)
    vol = ensure_vol_shape(vol, p1_vol_shape)
    np.save(kit_unvisible_view_mask_path, vol)
    if DEBUG:
        TSDFHelper.to_mesh(vol, kit_unvisible_view_mask_path.parent /
                           f"{kit_unvisible_view_mask_path.stem}.obj", voxel_size)


if __name__ == "__main__":
    main()
