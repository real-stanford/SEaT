
from pathlib import Path
from omegaconf import DictConfig
import numpy as np
import hydra
import torch
from utils import seed_all_int, ensure_vol_shape
from utils.metric import compute_iou, chamfer_distance
from real_world.utils import get_tn_bounds
from learning.dataset import ResultDataset
from learning.vol_match_rotate import VolMatchRotate
from learning.vol_match_transport import VolMatchTransport
from baseline.transportnet import Transportnet
from icecream import ic as print_ic
import h5py
import random

def prepare_sample(sample):
    p0_vol, p1_vol, p1_coords, p1_coords_user, p1_ori, concav_ori, sym = sample.values()
    p0_vol_rotate_ten = torch.tensor(p0_vol).unsqueeze(dim=0)
    p1_vol_ten = torch.tensor(p1_vol).unsqueeze(dim=0)
    user_coords_gt_ten = torch.tensor(p1_coords).unsqueeze(dim=0)
    user_coords_ten = torch.tensor(p1_coords_user).unsqueeze(dim=0)
    p1_ori_gt_ten = torch.tensor(p1_ori).unsqueeze(dim=0)
    batch = {
        "p0_vol": p0_vol_rotate_ten,
        "p1_vol": p1_vol_ten, 
        "p1_coords": user_coords_gt_ten,
        "p1_coords_user": user_coords_ten,
        "p1_ori": p1_ori_gt_ten,
        "concav_ori": torch.tensor([concav_ori]),
        "symmetry": torch.tensor([sym]),
    }
    return batch

def get_pt_from_vol(vol, n_pt = 1000):
    x, y, z = np.where(vol<=0)
    pts = np.array(list(zip(x,y,z)))
    sampled_inds = random.sample(range(pts.shape[0]), n_pt)
    sampled_pts = pts[sampled_inds, :]
    return sampled_pts

def get_preds(batch, transporter, rotator):
    with torch.no_grad():
        _, pred_coords, _, pos_diff = transporter.run(batch, training=False, log=False, calc_loss=True)
        batch['p1_coords'] = pred_coords.astype(int)
        _, _, pred_ori, rot_diff = rotator.run(batch, training=False, log=False, calc_loss=True)
        pred_coords = pred_coords[0]
    return pred_coords, pred_ori, pos_diff, rot_diff

def get_sc_data(dataset_root, vol_shape):
    data_paths = list(dataset_root.glob("**/data.hdf"))
    size = len(data_paths)
    print(f"Using SCDatsetRealworld from {dataset_root}, size={size}")
    gt_vols, pred_vols, gt_pcs, pred_pcs = [], [], [], []
    for index in range(size):
        hdf_path = data_paths[index]
        with h5py.File(str(hdf_path), "r") as hdf:
            gt_vol = ensure_vol_shape(np.array(hdf.get("gt_vol")), vol_shape)
            pred_vol = ensure_vol_shape(np.array(hdf.get("pred_vol")), vol_shape)
        gt_pc = get_pt_from_vol(gt_vol, n_pt = 10000)
        pred_pc = get_pt_from_vol(pred_vol, n_pt = 10000)
        gt_vols.append(gt_vol)
        pred_vols.append(pred_vol)
        gt_pcs.append(gt_pc)
        pred_pcs.append(pred_pc)
    return np.array(gt_vols), np.array(pred_vols), np.array(gt_pcs), np.array(pred_pcs)

def get_seg_data(dataset_root):
    data_paths = list(dataset_root.glob("**/data.hdf"))
    size = len(data_paths)
    print(f"Using SegDatsetRealworld from {dataset_root}, size={size}")
    gt_masks, pred_masks = [], []
    for index in range(size):
        hdf_path = data_paths[index]
        with h5py.File(str(hdf_path), "r") as hdf:
            gt_mask = np.array(hdf.get("gt_mask"))
            pred_mask = np.array(hdf.get("pred_mask"))
        gt_masks.append(gt_mask)
        pred_masks.append(pred_mask)
    return np.array(gt_masks), np.array(pred_masks)

def evaluate_seg(gt_mask, pred_mask):
    # mIoU, mAP
    iou = compute_iou(gt_mask, pred_mask)
    return iou

def evaluate_sc(gt_vols, pred_vols, gt_pcs, pred_pcs):
    # mIoU, chamfer L1
    iou = compute_iou(gt_vols, pred_vols)
    cdist = chamfer_distance(gt_pcs, pred_pcs)
    return iou, cdist

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    seed_all_int(100)
    SEG = True
    SC_OBJ = True
    SC_KIT = True
    SNAP = True

    voxel_size = cfg.env.voxel_size
    dataset_path =  Path('dataset/eval_realworld')
    
    # evaluate seg
    if SEG:
        seg_samples = get_seg_data(dataset_path / 'seg')
        seg_iou = evaluate_seg(*seg_samples)
        seg_miou = np.mean(seg_iou)
        print_ic(seg_miou)

    # import matplotlib.pyplot as plt
    # output_dir = Path('test')
    # output_dir.mkdir(exist_ok=True)
    # for i in range(len(seg_samples[0])):
    #     gt_mask, pred_mask = seg_samples[0][i], seg_samples[1][i]
    #     plt.imshow(gt_mask)
    #     plt.savefig(str(output_dir / f'{i}_gt.png'))
    #     plt.close()
    #     plt.imshow(pred_mask)
    #     plt.savefig(str(output_dir / f'{i}_pred_{seg_iou[i]:.3f}.png'))
    #     plt.close()
    # return
    
    # evaluate sc obj
    if SC_OBJ:
        sc_obj_samples = get_sc_data(dataset_path / 'sc_obj', vol_shape=(128,128,128))
        sc_obj_iou, sc_obj_cdist = evaluate_sc(*sc_obj_samples)
        sc_obj_miou = np.mean(sc_obj_iou)
        sc_obj_cdist_mean = np.mean(sc_obj_cdist) * voxel_size * 1000 # mm
        print_ic(sc_obj_miou, sc_obj_cdist_mean)

    # evaluate sc kit
    if SC_KIT:
        sc_kit_samples = get_sc_data(dataset_path / 'sc_kit', vol_shape=(400,400,256))
        sc_kit_iou, sc_kit_cdist = evaluate_sc(*sc_kit_samples)
        sc_kit_miou = np.mean(sc_kit_iou)
        sc_kit_cdist_mean = np.mean(sc_kit_cdist) * voxel_size * 1000 # mm
        print_ic(sc_kit_miou, sc_kit_cdist_mean)

    # evaluate snapnet
    def evaluate_snap(sample):
        sample_sc, sample_raw, sample_tn = sample[-3:]
        sample_sc_ten = prepare_sample(sample_sc)
        sample_raw_ten = prepare_sample(sample_raw)
        _, _, sc_pos_diff, sc_rot_diff = get_preds(sample_sc_ten, sc_transporter, sc_rotator)
        _, _, raw_pos_diff, raw_rot_diff = get_preds(sample_raw_ten, raw_transporter, raw_rotator)
        with torch.no_grad():
            _, _, (tn_pos_diff, tn_rot_diff) = transporternet.run(sample_tn, training=False)
        return sc_pos_diff, sc_rot_diff, raw_pos_diff, raw_rot_diff, tn_pos_diff, tn_rot_diff

    if SNAP:
        snap_dataset = ResultDataset.from_cfg(cfg, dataset_path / 'snap', realworld=True)
        voxel_size = cfg.env.voxel_size
        vm_cfg = cfg.vol_match_6DoF 
        gt_transport_path = 'checkpoints/oracle_transporter'
        gt_rotate_path = 'checkpoints/oracle_rotator'
        sc_transport_path = 'checkpoints/sc_transporter'
        sc_rotate_path = 'checkpoints/sc_rotator'
        raw_transport_path = 'checkpoints/raw_rotator'
        raw_rotate_path = 'checkpoints/raw_rotator'
        gt_transporter = VolMatchTransport.from_cfg(vm_cfg, voxel_size, load_model=True, log=False, model_path=gt_transport_path)
        gt_rotator = VolMatchRotate.from_cfg(vm_cfg, load_model=True, log=False, model_path=sc_rotate_path)
        sc_transporter = VolMatchTransport.from_cfg(vm_cfg, voxel_size, load_model=True, log=False, model_path=gt_transport_path)
        sc_rotator = VolMatchRotate.from_cfg(vm_cfg, load_model=True, log=False, model_path=sc_rotate_path)
        raw_transporter = VolMatchTransport.from_cfg(vm_cfg, voxel_size, load_model=True, log=False, model_path=raw_transport_path)
        raw_rotator = VolMatchRotate.from_cfg(vm_cfg, load_model=True, log=False, model_path=raw_rotate_path)
        transporternet = Transportnet.from_cfg(cfg.evaluate, load_model=True, view_bounds_info=get_tn_bounds())

        snap_diffs = []
        for snap_sample in snap_dataset:
            all_sample_sc, all_sample_raw, all_sample_tn = snap_sample[-3:]
            for sample in list(zip(all_sample_sc, all_sample_raw, all_sample_tn)):
                snap_diffs.append(evaluate_snap(sample))
        snap_diffs = np.array(snap_diffs)
        snap_diffs_med = np.median(snap_diffs, axis=0)
        sc_pos_med, sc_ori_med, raw_pos_med, raw_rot_med, tn_pos_med, tn_rot_med = snap_diffs_med
        print_ic(sc_pos_med, sc_ori_med, raw_pos_med, raw_rot_med, tn_pos_med, tn_rot_med)

if __name__ == "__main__":
    main()
