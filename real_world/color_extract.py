# Contains POC code for extracting mode/mean color for the volume
import sys
from pathlib import Path
root_path = Path(__file__).parent.parent.absolute()
# print("Adding to python path: ", root_path)
sys.path = [str(root_path)] + sys.path

import hydra
from omegaconf import DictConfig
import numpy as np
from learning.srg import SRG
from environment.real.cameras import RealSense
from real_world.utils import color_mask_rgb, get_obj_masks_tilted
from matplotlib import pyplot as plt


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Get the masks.
    # Using the mask find out the mean color excluding outliers

    # setup srg
    cropped_vol_shape = np.array(cfg.env.cropped_vol_shape)
    hw = np.ceil(cropped_vol_shape / 2).astype(np.int) # half width
    voxel_size = cfg.env.voxel_size
    print(cropped_vol_shape, voxel_size)
    srg = SRG(cfg.perception, hw)

    # setup camera
    bin_cam = RealSense()
    rgb, d = bin_cam.get_camera_data(avg_depth=True)

    # Get objects masks from tilted camera image
    masks = get_obj_masks_tilted(rgb, d, bin_cam.pose, bin_cam.color_intr, cfg, srg)
    # For each mask, show overlayed mask on rgb and a detected color patch
    for mask in masks:
        mask_rgb = color_mask_rgb(mask, rgb)
        color_patch = np.empty((100, 100, 3))
        for i in range(3):
            color_patch[:, :, i] = mask_rgb[i]
         
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(rgb, cmap='gray')
        ax[0].imshow(mask, cmap='jet', alpha=0.7) # interpolation='none'
        ax[1].imshow(color_patch)
        ax[1].set_title(f"RGB: {color_patch[0, 0, :]}")
        plt.show()


if __name__ == "__main__":
    main()