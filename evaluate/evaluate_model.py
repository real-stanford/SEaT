from learning.vol_match_rotate import VolMatchRotate
from baseline.transportnet import Transportnet
from environment.utils import SCENETYPE
import random
from matplotlib import pyplot as plt
from matplotlib import patches
from omegaconf import DictConfig
import torch
from pathlib import Path
from numpy.linalg import norm

from learning.dataset import SceneDatasetMaskRCNN, SceneDatasetShapeCompletion, SceneDatasetShapeCompletionSnap, VolMatchDataset, TNDataset
from learning.vol_match_transport import VolMatchTransport
from learning.vol_match_rotate import VolMatchRotate
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from utils import get_dataloader, get_device, init_ray, get_ray_fn, next_loop_iter, calcMIoU, seed_all_int
from utils.rotation import get_quat_diff
from environment.meshRendererEnv import dump_vol_render_gif
from sys import exit
import ray
from evaluate.html_vis import html_visualize, visualize_helper
from learning.seg import get_transform
import numpy as np
from vision_utils import utils
import os, sys

def setup_evaluate(cfg: DictConfig, model_path):
    device = get_device()
    use_ray = init_ray(cfg.ray)
    evaluate_dir = model_path.parent / f"evaluate_{model_path.stem}"
    if evaluate_dir.exists():
        print(f"{evaluate_dir} already exists. Exiting")
        exit()
    evaluate_dir.mkdir(parents=True)
    return device, use_ray, evaluate_dir

def dump_sc_model_preds_gifs(inp, target, pred, dump_dir, voxel_size, visualize_tsdf_gif):
    dump_dir.mkdir(exist_ok=True)
    dump_vol_render_gif(inp, dump_dir / "inp.obj", voxel_size,
                        visualize_mesh_gif=True, visualize_tsdf_gif=visualize_tsdf_gif)
    dump_vol_render_gif(target, dump_dir / "target.obj", voxel_size,
                        visualize_mesh_gif=True, visualize_tsdf_gif=visualize_tsdf_gif)
    dump_vol_render_gif(pred, dump_dir / "pred.obj", voxel_size,
                        visualize_mesh_gif=True, visualize_tsdf_gif=visualize_tsdf_gif)

def evaluate_sc_model(cfg: DictConfig):
    model_path = Path(cfg.evaluate.model_path)
    device = get_device()
    use_ray = init_ray(cfg.ray)
    evaluate_dir = Path(cfg.evaluate.save_path)
    evaluate_dir.mkdir(exist_ok=True, parents=True)

    voxel_size = cfg.env.voxel_size
    sc_model = torch.load(model_path, map_location=device).eval()

    scene_type = cfg.evaluate.scene_type
    # dataset = SceneDatasetShapeCompletion(Path(cfg.evaluate.dataset_path) / scene_type / cfg.evaluate.dataset_split, scene_type)
    dataset = SceneDatasetShapeCompletionSnap(Path(cfg.evaluate.dataset_path) / cfg.evaluate.dataset_split, scene_type)
    dataloader = DataLoader(dataset, batch_size=cfg.evaluate.batch_size,
                            num_workers=cfg.evaluate.num_workers, shuffle=True)

    num_samples = cfg.evaluate.num_samples
    curr_sample = 0
    pbar = tqdm(num_samples, desc="generating predictions", dynamic_ncols=True)
    tasks = list()
    dump_sc_model_preds_gifs_remote = ray.remote(dump_sc_model_preds_gifs)

    ids = list()
    cols = ["input", "target", "prediction"]
    data = dict()
    mIoU, total = 0, 0

    for inps, targets in dataloader:
    # for inps, targets, preds, labels in dataloader:
        with torch.no_grad():
            inps, targets = inps.to(device).float(), targets.to(device).float()
            preds = sc_model(inps)
            # preds = preds.to(device).float()

        inps = inps.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        mIoU += calcMIoU(preds, targets) * len(inps)
        total += len(inps)
        for i in range(len(inps)):
            # Generate visualizations
            label = curr_sample
            # label = labels[i].cpu().item()
            pred_dir = evaluate_dir / str(label)
            if use_ray:
                task = dump_sc_model_preds_gifs_remote.remote(
                    inps[i][0], targets[i][0], preds[i][0],
                    pred_dir, voxel_size, cfg.evaluate.dump_vols.tsdf_gifs
                )
            else:
                task = dump_sc_model_preds_gifs(
                    inps[i][0], targets[i][0], preds[i][0],
                    pred_dir, voxel_size, cfg.evaluate.dump_vols.tsdf_gifs
                )
            ids.append(str(label))
            data[f"{label}_input"] = str((pred_dir / "inp.gif").relative_to(evaluate_dir))
            data[f"{label}_target"] = str((pred_dir / "target.gif").relative_to(evaluate_dir))
            data[f"{label}_prediction"] = str((pred_dir / "pred.gif").relative_to(evaluate_dir))

            tasks.append(task)
            pbar.update(1)
            curr_sample += 1

            if curr_sample + 1 > num_samples:
                break
        if curr_sample + 1 > num_samples:
            break
    if use_ray:
        ray.get(tasks)
    pbar.close()

    html_visualize(evaluate_dir, data, ids, cols, title="Results")
    os.system(f'rm -rf {evaluate_dir}/**/*.obj {evaluate_dir}/**/*.urdf')
    print(f'mIoU={mIoU/total}')

def dump_seg_vis(img, boxes, target_boxes, scores, masks, target_masks, score_threshold, mask_threshold, log_path):
    """
        img: [h, w, c]
        boxes: [n, 4]
        scores: [n]
        masks: [n, h, w]
        target_masks: [n', h, w]
    """
    log_path.mkdir(parents=True, exist_ok=True)
    dump_paths = dict()
    _, ax = plt.subplots(1)
    valid_indices = np.where(scores >= score_threshold)[0]

    def plot_boxes(bxs, dump_path):
        ax.imshow(img)
        for box in bxs:
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2]-box[0], box[3]-box[1],
                linewidth=1, edgecolor='r', facecolor='none'
            ) 
            ax.add_patch(rect)
        plt.savefig(dump_path)
        plt.cla()
        return dump_path

    if len(target_boxes) != 0:
        dump_paths["gt_boxes"] = plot_boxes(target_boxes, log_path / "gt_boxes.png")
    if len(boxes) != 0:
        dump_paths["pred_boxes"] = plot_boxes(boxes[valid_indices], log_path / "pred_boxes.png")

    def plt_overlayed_mask(msk):
        mask = np.empty_like(msk)
        mask[msk < mask_threshold] = 0
        mask[msk >= mask_threshold] = 1
        mask *= 255
        # overlay with rgb image
        ax.imshow(img, cmap='gray')
        ax.imshow(mask, cmap='jet', alpha=0.5) # interpolation='none'

    for i in range(len(target_masks)):
        # ax.imshow(target_masks[i] * 255)
        plt_overlayed_mask(target_masks[i])
        dump_paths[f"gt_mask_{i}"] = log_path / f"gt_mask_{i}.png"
        plt.savefig(dump_paths[f"gt_mask_{i}"])
        plt.cla()

    if len(masks) != 0:
        for i, ind in enumerate(valid_indices):
            # ax.imshow(masks[ind] * 255)
            plt_overlayed_mask(masks[ind])
            ax.set_title(f"Score: {scores[ind]}")
            dump_paths[f"pred_mask_{i}"] = log_path / f"pred_mask_{i}.png"
            plt.savefig(dump_paths[f"pred_mask_{i}"])
            plt.cla()

    plt.close()
    return dump_paths

def evaluate_seg(cfg: DictConfig):
    model_path = Path(cfg.evaluate.model_path)
    device, use_ray, evaluate_dir = setup_evaluate(cfg, model_path)

    use_depth = cfg.evaluate.use_depth
    normalize_depth = cfg.evaluate.normalize_depth
    val_transforms = get_transform(True, use_depth, normalize_depth)
    model = torch.load(model_path, map_location=device)
    dataset_path = Path(cfg.evaluate.dataset_path)
    dataset = SceneDatasetMaskRCNN(dataset_path, use_depth, val_transforms)
    indices = range(len(dataset))
    if cfg.evaluate.num_samples != -1:    
        indices = random.sample(indices, min(len(indices), cfg.evaluate.num_samples))
    dataset = Subset(dataset, indices)
    dataloader = DataLoader(
        dataset, shuffle=True, num_workers=cfg.evaluate.num_workers,
        batch_size=cfg.evaluate.batch_size, collate_fn = utils.collate_fn)

    dump_seg_vis_remote = ray.remote(dump_seg_vis)
    tasks = list()
    datapoint_counter = 0
    for imgs, targets in dataloader:
        imgs = list(image.to(device) for image in imgs)
        outputs = model.forward(imgs)
        imgs = [img.detach().permute(1, 2, 0).cpu().numpy() for img in imgs]
        # boxes, scores, masks, target_masks, score_threshold, log_path):
        batch_boxes = [output["boxes"].detach().cpu().numpy()
                       for output in outputs]
        batch_target_boxes = [target["boxes"].detach().cpu().numpy()
                       for target in targets]
        batch_scores = [output["scores"].detach().cpu().numpy()
                        for output in outputs]
        batch_masks = [output["masks"].squeeze(
            dim=1).detach().cpu().numpy() for output in outputs]
        batch_target_masks = [target["masks"].squeeze(
            dim=1).numpy() for target in targets]

        if use_ray: 
            tasks += [
                dump_seg_vis_remote.remote(
                    img, boxes, target_boxes, scores, masks, target_masks,
                    cfg.perception.seg.mask_score_threshold, cfg.perception.seg.mask_threshold,
                    evaluate_dir / str(datapoint_counter + i)
                )
                for i, (img, boxes, target_boxes, scores, masks, target_masks) in
                enumerate(zip(imgs, batch_boxes, batch_target_boxes, batch_scores, batch_masks, batch_target_masks))
            ]
        else:
            tasks += [
                dump_seg_vis(
                    img, boxes, scores, masks, target_masks,
                    cfg.perception.seg.mask_score_threshold, cfg.perception.seg.mask_threshold,
                    evaluate_dir / str(datapoint_counter + i)
                )
                for i, (img, boxes, scores, masks, target_masks) in
                enumerate(zip(imgs, batch_boxes, batch_scores,
                              batch_masks, batch_target_masks))
            ]
        datapoint_counter += len(imgs) 
    if use_ray:
        tasks = ray.get(tasks)

    # generate webpage now
    # - now I have the dump paths. What next?
    data = dict()
    ids = list()
    cols = list()
    cols += [f"gt_mask_{i}" for i in range(10)]
    cols += ["gt_boxes", "pred_boxes"]
    cols += [f"pred_mask_{i}" for i in range(10)]
    for id, dump_paths in enumerate(tasks):
        ids.append(str(id))
        for col, path in dump_paths.items():
            data[f"{id}_{col}"] = str(path.relative_to(evaluate_dir))
    html_visualize(str(evaluate_dir), data, ids,
                   cols, title=f"Seg Predictions {model_path} {dataset_path}")

def evaluate_tn_model(cfg: DictConfig):

    dataset = TNDataset.from_cfg(cfg.evaluate, 'val')
    model = Transportnet.from_cfg(cfg.evaluate, load_model=True)
    evaluate_size = min(cfg.evaluate.evaluate_size, len(dataset))
    output_dir = Path(cfg.evaluate.evaluate_save_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    use_ray = init_ray(cfg.ray)
    fn = get_ray_fn(cfg.ray, TNDataset.visualize_tn)
    tasks = list()
    total_pos_diff, total_ori_diff = [], []

    for num in range(evaluate_size):
        sample = dataset[num]
        with torch.no_grad():
            _, (pred_pos, pred_quat), (pos_diff, quat_diff) = model.run(sample, training=False)
        cmap_obj, _, cmap_kit, _, pick_pos, place_pos, place_ori, _, syms, _, _ = sample
        gt = (pick_pos, place_pos, place_ori)
        pred = (pred_pos, pred_quat)
        tasks.append(fn(output_dir, num, dataset.view_bounds_obj, dataset.view_bounds_kit, dataset.pix_size, cmap_obj, cmap_kit, syms, gt, pred, (pos_diff, quat_diff)))
        # print(pos_diff, quat_diff)
        total_pos_diff.append(pos_diff)
        total_ori_diff.append(quat_diff)
    total_pos_diff, total_ori_diff = np.array(total_pos_diff), np.array(total_ori_diff)
    pos_med, pos_mean, pos_max = np.median(total_pos_diff), np.mean(total_pos_diff), np.max(total_pos_diff)
    ori_med, ori_mean, ori_max = np.median(total_ori_diff), np.mean(total_ori_diff), np.max(total_ori_diff)
    eval_str = f'size: {evaluate_size}, med_pos_diff: {pos_med:.3f} mm, mean_pos_diff: {pos_mean:.3f} mm, max_pos_diff: {pos_max:.3f} mm '
    eval_str +=  f'med_ori_diff: {ori_med:.3f} deg, mean_ori_diff: {ori_mean:.3f} deg, max_ori_diff: {ori_max:.3f}'
    print(eval_str)
    if use_ray:
        ray.get(tasks)
    cols = ["gt_pick", "gt_place", "gt_overlay", "gt_ori_z", "pred_pick", "pred_place", "pred_overlay", "pred_ori_z", "diff", "symmetry"]
    visualize_helper(tasks, output_dir, cols)

def evaluate_vol_match_model_robustness(cfg: DictConfig):
    # Load model
    vm_cfg = cfg.vol_match_6DoF
    voxel_size = cfg.env.voxel_size
    kit_shape = np.array(vm_cfg.p1_vol_shape)
    transporter = VolMatchTransport.from_cfg(vm_cfg, cfg.env.voxel_size, load_model=True, log=False) 
    rotator = VolMatchRotate.from_cfg(vm_cfg, load_model=True, log=False) 
    n = 20
    output_dir = Path(f'val_ori_rob_10')
    output_dir.mkdir(exist_ok=True, parents=True)

    all_logs = []

    use_gt_ori = False
    use_gt_pos = True

    for i in range(8):
        dataset = VolMatchDataset.from_cfg(cfg, Path(vm_cfg.dataset_path) / vm_cfg.dataset_split)
        dataset.min_d = 10
        dataset.min_angle = i * 0.06
        dataset.use_gt_ori = False
        dataloader = get_dataloader(dataset, seed=cfg.seeds.test, batch_size=1, shuffle=False, num_workers=1)
        dataloader_iter = iter(dataloader)
        p_diffs, ori_diffs = [], []
        gt_p_diffs, gt_ori_diffs = [], []
        for ind in tqdm(range(n), dynamic_ncols=True):
            dataloader_iter, batch = next_loop_iter(dataloader_iter, dataloader)
            p0_vol, p1_vol, p1_coords, p1_coords_user, p1_ori, concav_ori, syms = batch.values()
            with torch.no_grad():
                if use_gt_pos:
                    pred_coords, p_diff = p1_coords_user.cpu().numpy(), 0
                    batch['p1_coords'] = torch.tensor(pred_coords).int()
                else:
                    _, pred_coords, _, p_diff = transporter.run(batch, training=False, log=False, calc_loss=False)
                    batch['p1_coords'] = torch.tensor(pred_coords).int()
                if use_gt_ori:
                    pred_ori, ori_diff = p1_ori[0].cpu().numpy(), 0
                else:
                    _, _, pred_ori, ori_diff = rotator.run(batch, training=False, log=False, calc_loss=False)
                p_diffs.append(p_diff)
                ori_diffs.append(ori_diff)
                gt_p_diff_vec = p1_coords[0].cpu().numpy()-p1_coords_user[0].cpu().numpy()
                gt_p_diff = norm(gt_p_diff_vec) * voxel_size * 1000
                gt_ori_diff = get_quat_diff(p1_ori[0].cpu().numpy(), np.array([0,0,0,1])) * 180/np.pi
                gt_p_diffs.append(gt_p_diff)
                gt_ori_diffs.append(gt_ori_diff)
        def stats(data, name, unit):
            mean, median, ma = np.mean(data), np.median(data), np.max(data)
            plt.hist(data,  bins='auto')
            plt.xlabel(f'{name} ({unit})')
            plt.savefig(output_dir / f'{name}_hist.jpg')
            plt.close()
            print(f'{name} diff ({unit}): mean {mean:.3f}, median {median:.3f}, ma: {ma:.3f}')
        if not vm_cfg.use_gt_pos:
            stats(gt_p_diffs, 'gt_position', 'mm')
            stats(p_diffs, 'position', 'mm')
        if not vm_cfg.use_gt_ori:
            stats(gt_ori_diffs, 'gt_orientation', 'deg')
            stats(ori_diffs, 'orientation', 'deg')
        all_logs.append([p_diffs, ori_diffs, gt_p_diffs, gt_ori_diffs])
    all_logs = np.array(all_logs)
    np.save(output_dir / 'diffs.npy', all_logs)

def evaluate_vol_match_model(cfg: DictConfig):
    # Load model
    vm_cfg = cfg.vol_match_6DoF
    voxel_size = cfg.env.voxel_size
    kit_shape = np.array(vm_cfg.p1_vol_shape)
    dataset = VolMatchDataset.from_cfg(cfg, Path(vm_cfg.dataset_path) / vm_cfg.dataset_split)
    dataloader = get_dataloader(dataset, seed=cfg.seeds.test, batch_size=1, shuffle=False, num_workers=1)
    dataloader_iter = iter(dataloader)
    transporter = VolMatchTransport.from_cfg(vm_cfg, cfg.env.voxel_size, load_model=True, log=False) 
    rotator = VolMatchRotate.from_cfg(vm_cfg, load_model=True, log=False) 
    n = min(vm_cfg.evaluate_size, len(dataset))
    output_dir = Path(f'{vm_cfg.evaluate_save_path}_{n}')
    output_dir.mkdir(exist_ok=True, parents=True)

    if vm_cfg.evaluate_gen_gifs:
        use_ray = init_ray(cfg.ray)
        fn = get_ray_fn(cfg.ray, VolMatchDataset.visualize_6dof, gpu_frac=1)
        tasks = list()

    p_diffs, ori_diffs = [], []
    gt_p_diffs, gt_ori_diffs = [], []
    for ind in tqdm(range(n), dynamic_ncols=True):
        dataloader_iter, batch = next_loop_iter(dataloader_iter, dataloader)
        p0_vol, p1_vol, p1_coords, p1_coords_user, p1_ori, concav_ori, syms = batch.values()
        with torch.no_grad():
            if vm_cfg.use_gt_pos:
                pred_coords, p_diff = p1_coords.cpu().numpy(), 0
            else:
                _, pred_coords, _, p_diff = transporter.run(batch, training=False, log=False, calc_loss=False)
                batch['p1_coords'] = torch.tensor(pred_coords).int()
            if vm_cfg.use_gt_ori:
                pred_ori, ori_diff = p1_ori[0].cpu().numpy(), 0
            else:
                _, _, pred_ori, ori_diff = rotator.run(batch, training=False, log=False, calc_loss=False)
            p_diffs.append(p_diff)
            ori_diffs.append(ori_diff)
            gt_p_diff_vec = p1_coords[0].cpu().numpy()-p1_coords_user[0].cpu().numpy()
            gt_p_diff = norm(gt_p_diff_vec) * voxel_size * 1000
            gt_ori_diff = get_quat_diff(p1_ori[0].cpu().numpy(), np.array([0,0,0,1])) * 180/np.pi
            gt_p_diffs.append(gt_p_diff)
            gt_ori_diffs.append(gt_ori_diff)
            # print(p_diff, ori_diff, gt_p_diff, gt_ori_diff)
            # print(f'Test {ind}: pos_diff: {p_diff:.3f} mm, ori_diff: {ori_diff:.3f} deg')
        if vm_cfg.evaluate_gen_gifs:
            tasks.append(fn(output_dir, ind, voxel_size, kit_shape, syms[0],
                        p0_vol[0], p1_vol[0], p1_coords[0], p1_coords_user[0], p1_ori[0], 
                        pred_coords[0], pred_ori, p_diff, ori_diff))

    log_data = np.array([p_diffs, ori_diffs, gt_p_diffs, gt_ori_diffs])
    def stats(data, name, unit):
        mean, median, ma = np.mean(data), np.median(data), np.max(data)
        plt.hist(data,  bins='auto')
        plt.xlabel(f'{name} ({unit})')
        plt.savefig(output_dir / f'{name}_hist.jpg')
        plt.close()
        print(f'{name} diff ({unit}): mean {mean:.3f}, median {median:.3f}, ma: {ma:.3f}')
    if not vm_cfg.use_gt_pos:
        stats(gt_p_diffs, 'gt_position', 'mm')
        stats(p_diffs, 'position', 'mm')
    if not vm_cfg.use_gt_ori:
        stats(gt_ori_diffs, 'gt_orientation', 'deg')
        stats(ori_diffs, 'orientation', 'deg')
    np.save(output_dir / f'diffs.npy', log_data)
    
    if vm_cfg.evaluate_gen_gifs and use_ray:
        tasks = ray.get(tasks)
    if vm_cfg.evaluate_gen_gifs:
        cols = ["data_vis", "gt_vis", 'gt', 'symmetry', 'pred_vis', 'diff']
        visualize_helper(tasks, output_dir, cols)
        # remove obj files for faster download
        os.system(f'rm -rf {vm_cfg.evaluate_save_path}/*.obj {vm_cfg.evaluate_save_path}/*.urdf')