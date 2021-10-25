from environment.baseEnv import BaseEnv
from omegaconf import DictConfig
from pathlib import Path
import h5py
from PIL import Image
import numpy as np
import omegaconf
import ray
from utils import get_split_file, get_split_obj_roots, init_ray, get_ray_fn, get_crop
from utils.rotation import quat_to_euler, sample_rot, get_quat_diff, get_quat_diff_sym, sample_rot_roll
from environment.meshRendererEnv import MeshRendererEnv, dump_tsdf_vis, dump_vol_render_gif
import random
import numpy as np
from omegaconf import DictConfig
from matplotlib import pyplot as plt
from matplotlib import patches
from evaluate.html_vis import visualize_helper
from learning.dataset import TNDataset, VolMatchDataset
from learning.dataset import Depth2OrientDataset
from scipy.ndimage import rotate
from random import shuffle
import pybullet as p
from environment.camera import SimCameraPosition
from environment.utils import get_body_colors, set_visible
from PIL import Image
from data_generation import get_seg_sc_dataset_path


def generate_data_visualizations(data_dir: Path, data_cfg: DictConfig, voxel_size: float, cropped_vol_shape) -> None:
    with h5py.File(data_dir / "data.hdf", "r") as hdf:
        dump_paths = dict()
        rgb = np.array(hdf.get("rgb"))
        Image.fromarray(rgb).save(data_dir / "rgb.png")
        d = np.array(hdf.get("d"))
        d = ((d - d.min()) * 255 / (d.max() - d.min())).astype(np.uint8)
        Image.fromarray(d).save(data_dir / "d.png")
        masks = (np.array(hdf.get("masks")) * 255).astype(np.uint8)
        for i in range(len(masks)):
            Image.fromarray(masks[i]).save(data_dir / f"masks_{i}.png")

        boxes = np.array(hdf.get("boxes"))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(rgb)
        for box_index, box in enumerate(boxes):
            width = box[2]-box[0]
            height = box[3]-box[1]
            if width == 0 or height == 0:
                print(f"[Error ({data_dir})] width or height is zero for box {box_index}")
            rect = patches.Rectangle(
                (box[0], box[1]),
                width, height,
                linewidth=1, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
        plt.savefig(data_dir / "rgb_boxes.png")
        plt.close(fig)
        dump_paths["rgb_boxes"] = data_dir / "rgb_boxes.png"

        # volumes
        # ok. Also need to dump the cropped volumes inp and outputs
        if data_cfg.dump_vols.obj:
            inps = np.array(hdf.get("sc_inps"))
            targets = np.array(hdf.get("sc_targets"))
            for idx, (inp, target) in enumerate(zip(inps, targets)):
                if (inp.shape != cropped_vol_shape).any():
                    print(f"[Error ({data_dir})] input shape {inp.shape} != cropped_vol_shape {cropped_vol_shape}\
                        at cropped_vol_indices {idx}")
                vol_gif_path, tsdf_gif_path = dump_vol_render_gif(
                    inp, data_dir / f"vols_{idx}_cropped_inp.obj", voxel_size,
                    visualize_mesh_gif=data_cfg.dump_vols.obj_gifs,
                    visualize_tsdf_gif=data_cfg.dump_vols.tsdf_gifs)
                if vol_gif_path is not None:
                    dump_paths[f"inp_vol_{idx}"] = vol_gif_path
                if tsdf_gif_path is not None:
                    dump_paths[f"inp_tsdf_{idx}_0"] = tsdf_gif_path[0]
                    dump_paths[f"inp_tsdf_{idx}_1"] = tsdf_gif_path[1]
                if (target.shape != cropped_vol_shape).any():
                    print(f"[Error ({data_dir})] target shape {target.shape} != cropped_vol_shape {cropped_vol_shape}\
                        at cropped_vol_indices {idx}")
                vol_gif_path, tsdf_gif_path = dump_vol_render_gif(
                    target, data_dir / f"vols_{idx}_cropped_target.obj", voxel_size,
                    visualize_mesh_gif=data_cfg.dump_vols.obj_gifs,
                    visualize_tsdf_gif=data_cfg.dump_vols.tsdf_gifs)
                if vol_gif_path is not None:
                    dump_paths[f"target_vol_{idx}"] = vol_gif_path
                if tsdf_gif_path is not None:
                    dump_paths[f"target_tsdf_{idx}_0"] = tsdf_gif_path[0]
                    dump_paths[f"target_tsdf_{idx}_1"] = tsdf_gif_path[1]
        return dump_paths

def evaluate_data(cfg: DictConfig):
    # Go inside dataset directory and generate visualizations for each
    scene_type = cfg.evaluate.scene_type
    if scene_type == 'kit':
        cropped_vol_shape = np.array(cfg.env.kit_vol_shape)
    elif scene_type == 'object':
        cropped_vol_shape = np.array(cfg.env.obj_vol_shape)
    else:
        print(f'Scene type not handled: {scene_type}')
        raise NotImplementedError

    use_ray = init_ray(cfg.ray)
    generate_data_visualizations_remote = ray.remote(
        generate_data_visualizations)
    tasks = list()
    dataset_path = get_seg_sc_dataset_path(cfg.evaluate)
    cols = ["rgb_boxes"]
    for i in range(8):
        cols += [f"inp_vol_{i}", f"inp_tsdf_{i}_0", f"inp_tsdf_{i}_1",
                 f"target_vol_{i}", f"target_tsdf_{i}_0", f"target_tsdf_{i}_1"]

    data_dirs = [
        data_dir
        for data_dir in dataset_path.iterdir()
        if data_dir.is_dir() and not data_dir.name.startswith(".")
    ]
    if cfg.evaluate.num_samples != -1:
        data_dirs = random.sample(data_dirs, min(len(data_dirs), cfg.evaluate.num_samples))

    data_dirs = random.sample(data_dirs, min(len(data_dirs), cfg.evaluate.num_samples))
    for data_dir in data_dirs:
        if use_ray:
            task = generate_data_visualizations_remote.remote(
                data_dir, cfg.evaluate, cfg.env.voxel_size, cropped_vol_shape)
        else:
            task = generate_data_visualizations(
                data_dir, cfg.evaluate, cfg.env.voxel_size, cropped_vol_shape)
        tasks.append(task)
    if use_ray:
        tasks = ray.get(tasks)

    visualize_helper(tasks, dataset_path, cols)

def evaluate_data_tn(cfg: DictConfig):
    use_ray = init_ray(cfg.ray)
    dataset = TNDataset.from_cfg(cfg.evaluate) 

    fn = get_ray_fn(cfg.ray, dataset.visualize_tn)
    tasks = list()
    indices = random.sample(range(len(dataset)), min(len(dataset), cfg.evaluate.evaluate_size))
    for i in indices:
        output_dir = dataset.data_paths[i].parent
        tasks.append(fn(output_dir, *dataset[i]))
    if use_ray:
        tasks = ray.get(tasks)
    
    cols = ["cmap", "cmap_overlay", "ori", "z"]
    dataset_path = Path('baseline/dataset/transporter/train')
    visualize_helper(tasks, dataset_path, cols)

def evaluate_data_depth2orient(cfg: DictConfig):
    use_ray = init_ray(cfg.ray)
    dataset_path = Path('dataset/vol_match_abc/val')
    dataset = Depth2OrientDataset(dataset_path) 

    fn = get_ray_fn(cfg.ray, Depth2OrientDataset.visualize_depth2orient)
    tasks = list()
    indices = random.sample(range(len(dataset)), min(len(dataset), cfg.evaluate.evaluate_size))
    for i in indices:
        output_dir = dataset.data_paths[i].parent
        tasks.append(fn(output_dir, *dataset[i]))
    if use_ray:
        tasks = ray.get(tasks)
    
    cols = ["part_img", "kit_img", "part_img_rot", "overlay", "ori", "symmetry"]
    visualize_helper(tasks, dataset_path, cols)

def evaluate_vol_match(cfg: DictConfig):
    use_ray = init_ray(cfg.ray)
    vm_cfg = cfg.vol_match_6DoF
    dataset = VolMatchDataset.from_cfg(cfg, Path(vm_cfg.dataset_path) / vm_cfg.dataset_split, 
                                        vol_type=None)
    voxel_size = float(cfg.env.voxel_size)
    kit_shape = np.array(vm_cfg.p1_vol_shape)
    try:
        gpu_frac = cfg.vol_match_6DoF.gpu_frac
        print("Using gpu_frac: ", gpu_frac)
    except Exception as e:
        gpu_frac = None
    fn = get_ray_fn(cfg.ray, VolMatchDataset.visualize_6dof, gpu_frac=gpu_frac)

    tasks = list()
    indices = random.sample(range(len(dataset)), min(len(dataset), 10
    ))

    # vic_quats, _ = sample_rot_roll(15/180*np.pi,0.1,10/180*np.pi)
    # print(len(vic_quats))
    # gt_diffs = []
    for ind in indices:
        output_dir = dataset.data_paths[ind].parent
        p0_vol, p1_vol, p1_coords, p1_coords_user, p1_ori, concav_ori, sym = dataset[ind].values()
        # input_diff = get_quat_diff(p1_ori, np.array([0,0,0,1])) * 180/np.pi
        # gt_diffs.append(input_diff)
        # diffs = get_quat_diff(vic_quats, p1_ori)
        # min_quat = vic_quats[np.argmin(diffs)]
        # diff = min(diffs)*180/np.pi
        # print(input_diff, diff)
        tasks.append(fn(output_dir, ind, voxel_size, kit_shape, sym,
                        p0_vol, p1_vol, p1_coords, p1_coords_user, p1_ori, gui=False))
    # gt_diffs = np.array(gt_diffs)
    # print(np.median(gt_diffs))
    if use_ray:
        tasks = ray.get(tasks)
    
    cols = ["data_vis", "gt_vis", 'gt']
    dataset_path = Path(dataset.dataset_root)
    visualize_helper(tasks, dataset_path, cols)

def visualize_obj_kit_placement(obj_path: Path, kit_path: Path):
    env = MeshRendererEnv(gui=False)
    p.setGravity(0, 0, 0)
    dump_path = {"id": obj_path.parent.name}
    obj_body_id = env.load_mesh(obj_path, urdf_path=obj_path.parent / f"{obj_path.name}_only_vm.urdf", rgba=[0, 1, 0, 1])
    # Capture image from bottom:
    obj_camera = SimCameraPosition([0, 0, -0.1], [0, 0, 0], [1, 0, 0])
    obj_rgb, _, _ = obj_camera.get_image()
    obj_rgb_path = obj_path.parent / f"{obj_path.stem}_obj.png"
    Image.fromarray(obj_rgb).save(obj_rgb_path)
    dump_path["obj"] = obj_rgb_path
    obj_visual_data = {obj_body_id: get_body_colors(obj_body_id)}
    set_visible(obj_visual_data, visible=False)

    env.load_mesh(kit_path, urdf_path=kit_path.parent / f"{kit_path.name}_only_vm.urdf", rgba=[1, 0, 0, 0.6])
    kit_camera = SimCameraPosition([0, 0, 0.1], [0, 0, 0], [1, 0, 0])
    kit_rgb, _, _ = kit_camera.get_image()
    kit_rgb_path = obj_path.parent / f"{obj_path.stem}_kit.png"
    Image.fromarray(kit_rgb).save(kit_rgb_path)
    dump_path["kit"] = kit_rgb_path

    set_visible(obj_visual_data, visible=True)
    gif_path = env.render(obj_path.parent / f"{obj_path.stem}.gif", bb_min=[-0.05, -0.05, -0.05], bb_max=[0.05, 0.05, 0.05])
    dump_path["obj_kit"] = gif_path
    
    return dump_path

def evaluate_prepared_data(cfg: DictConfig):
    # Ok.  Cool. What's the big idea?
    # Now I want to generate evaluations
    evaluate_size = cfg.evaluate.num_samples
    dataset_path = Path(cfg.evaluate.path)
    paths = list()
    for obj_path in dataset_path.rglob("**/obj.obj"):
        kit_path = obj_path.parent / f"kit_parts/kit.obj"
        paths.append((obj_path, kit_path))
    shuffle(paths)
    paths = paths[:evaluate_size]

    use_ray = init_ray(cfg.ray)
    fn = get_ray_fn(cfg.ray, visualize_obj_kit_placement, gpu_frac=cfg.evaluate.gpu_frac)
    tasks = list()
    for obj_path, kit_path in paths:
        tasks.append(fn(obj_path, kit_path))
    if use_ray:
        tasks = ray.get(tasks)
    
    # Now visualize using html viewer
    cols = ["id", "obj", "kit", "obj_kit"]
    html_file_name = "index.html"
    visualize_helper(tasks, dataset_path, cols, html_file_name=html_file_name)
