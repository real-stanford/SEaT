# Code for visualizing the segmentation mask. And getting their coordinates in the original space
import sys
from pathlib import Path
root_path = Path(__file__).parent.parent.absolute()
# print("Adding to python path: ", root_path)
sys.path = [str(root_path)] + sys.path

import hydra
from omegaconf import DictConfig
from environment.real.cameras import RealSense
from real_world.rw_utils import get_crops_wb, get_obj_bounds_mask, get_kit_bounds_mask
import numpy as np
from evaluate.evaluate_model import dump_seg_vis
from learning.srg import SRG
from matplotlib import pyplot as plt, use
from scipy.ndimage import rotate
from utils.ravenutils import np_unknown_cat
from shutil import rmtree


def vis_seg_crop(cfg: DictConfig):
    bin_cam = RealSense()
    rgb, d = bin_cam.get_camera_data(avg_depth=True)
    obj_crop, kit_crop, obj_crop_depth, kit_crop_depth, obj_center, kit_center, obj_angle, kit_angle = get_crops_wb(
        rgb, d, bin_cam.color_intr, bin_cam.pose, wb=None)

    # Load segmentation model and get prediction
    # get the basic environment running
    cropped_vol_shape = np.array(cfg.env.cropped_vol_shape)
    hw = np.ceil(cropped_vol_shape / 2).astype(np.int) # half width
    srg = SRG(cfg.perception, hw)
    masks, labels, scores = srg.get_seg(obj_crop)
    print(masks.shape)
    # Interesting. Let's see what can be done now?
    # I want to visualize all these masks somehow. But how? 

    mask_score_threshold = cfg.perception.seg.mask_score_threshold
    mask_threshold = cfg.perception.seg.mask_threshold
    log_path = Path("/tmp/real_pred")
    log_path.mkdir(exist_ok=True)
    dump_seg_vis(obj_crop, [], [], scores, masks, [], mask_score_threshold, mask_threshold, log_path)

    # For each mask, I want to identify the mask in the original image.
    # Cool. Just rotate back the mask 
    fig = plt.figure()

    def plt_overlayed_mask(msk, img, score):
        mask = np.empty_like(msk)
        mask[msk < mask_threshold] = 0
        mask[msk >= mask_threshold] = 1
        mask *= 255
        # overlay with rgb image
        plt.imshow(img, cmap='gray')
        plt.imshow(mask, cmap='jet', alpha=0.7) # interpolation='none'
        plt.set_title(f"Score: {scores}")

    for i, (mask, score) in enumerate(zip(masks, scores)):
        if score < mask_score_threshold:
            continue
        mask[mask>mask_threshold] = 1
        mask[mask<=mask_threshold] = 0

        rotate_mask = rotate(mask, -1 * obj_angle * 180 / np.pi, reshape=False)
        transform_mask = np.zeros_like(d)
        hh, hw = np.floor(np.array(rotate_mask.shape) / 2).astype(np.int)
        transform_mask[obj_center[1] - hh: obj_center[1] + hh, obj_center[0] - hw: obj_center[0] + hw] = rotate_mask[:2*hh, :2*hw]
        plt_overlayed_mask(transform_mask, rgb, score)
        plt.savefig(log_path / f"tran_mask_{i}.png")
        plt.clf()
    plt.close()
    

def vis_seg_masked(cfg: DictConfig):
    # setup camera 
    cam = RealSense()
    rgb, d = cam.get_camera_data()

    obj_mask = get_obj_bounds_mask(cam.pose, cam.color_intr, d.shape)
    rgb_masked_obj = rgb * np.stack((obj_mask,obj_mask,obj_mask), axis=-1).astype(np.int)
    kit_mask = get_kit_bounds_mask(cam.pose, cam.color_intr, d.shape)
    rgb_masked_kit = rgb * np.stack((kit_mask,kit_mask,kit_mask), axis=-1).astype(np.int)

    # def plt_mask(mask, ax):
    #     mask = np.copy(mask) * 255
    #     # overlay with rgb image
    #     ax.imshow(rgb, cmap='gray')
    #     ax.imshow(mask, cmap='jet', alpha=0.7) # interpolation='none'
    # _, ax = plt.subplots(2,2) 
    # plt_mask(obj_mask, ax[0, 0])
    # ax[0, 1].imshow(rgb_masked_obj)
    # plt_mask(kit_mask, ax[1, 0])
    # ax[1, 1].imshow(rgb_masked_kit)
    # plt.show()

    # Load segmentation model
    cropped_vol_shape = np.array(cfg.env.cropped_vol_shape)
    hw = np.ceil(cropped_vol_shape / 2).astype(np.int) # half width
    srg = SRG(cfg.perception, hw)
    masks, labels, scores = srg.get_seg(rgb)
    print(masks.shape)

    # Ok. Let's process all masks. Again
    # And ignore masks that are not objects
    mask_score_threshold = cfg.perception.seg.mask_score_threshold
    mask_threshold = cfg.perception.seg.mask_threshold

    log_path = Path("/tmp/real_pred_all")
    if log_path.exists():
        rmtree(log_path)
    log_path.mkdir()
    dump_seg_vis(rgb, [], [], scores, masks, [], mask_score_threshold, mask_threshold, log_path)

    log_path = Path("/tmp/real_pred")
    if log_path.exists():
        rmtree(log_path)
    log_path.mkdir()
    tms = None
    tss = None
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if score < mask_score_threshold:
            continue
        tm =  mask * obj_mask
        if np.any(tm > mask_threshold):
            tms = np_unknown_cat(tms, tm)
            tss = np_unknown_cat(tss, score)
    dump_seg_vis(rgb, [], [], tss, tms, [], mask_score_threshold, mask_threshold, log_path)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # vis_seg_crop(cfg)
    vis_seg_masked(cfg)

if __name__ == "__main__":
    main()
