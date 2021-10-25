"""
This file contains code for collecting and labeling real world dataset
"""
import shutil
import sys
from pathlib import Path
root_path = Path(__file__).parent.parent.absolute()
# print("Adding to python path: ", root_path)
sys.path = [str(root_path)] + sys.path

import hydra
from omegaconf import DictConfig
from matplotlib import pyplot as plt
import numpy as np
from environment.real.cameras import RealSense
from utils import mkdir_fresh

def collect_images():
    DUMP_PNG_IMGS = True
    bin_cam = RealSense()
    dataset_root = Path("real_world/dataset/")
    if not dataset_root.exists():
        dataset_root.mkdir()
    # else:
    #     print("Datset directory already exist. Please rename/delete it and run again.")

    def save_img(path: Path, img_arr: np.ndarray):
        np.save(path, img_arr) 
        if DUMP_PNG_IMGS:
            plt.imsave(path.parent / f"{path.stem}.png", img_arr)
            
    i = 0
    np.save(dataset_root / "camera_pose.npy", bin_cam.pose)
    np.save(dataset_root / "camera_depth_scale.npy", bin_cam.depth_scale)
    np.save(dataset_root / "camera_color_intr.npy", bin_cam.color_intr)
    np.save(dataset_root / "camera_depth_intr.npy", bin_cam.depth_intr)
    kit_bounds_mask_filename = "kit_bounds_mask.npy"
    shutil.copyfile(
        f"real_world/{kit_bounds_mask_filename}",
        dataset_root / kit_bounds_mask_filename
    )
    while True:
        input("Setup scene and press enter ...")
        output_dir = mkdir_fresh(dataset_root / f"{i}")
        rgb, d = bin_cam.get_camera_data(avg_depth=True, avg_over_n=50)
        save_img(output_dir / "rgb.npy", rgb)
        save_img(output_dir / "d.npy", d)
        i += 1

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    collect_images()


if __name__ == "__main__":
    main()
