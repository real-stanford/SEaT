
from operator import mul
from utils.tsdfHelper import TSDFHelper
from utils.rotation import invert_quat, normal_to_quat, quat_to_euler, get_quat_diff, multiply_quat, invert_quat, uniform_sample_quaternion
from utils.ravenutils import np_unknown_cat, transform_pointcloud, get_heightmap, get_pointcloud
from real_world.rw_utils import get_tn_bounds, get_intrinsics
from environment.meshRendererEnv import MeshRendererEnv
from pathlib import Path
import numpy as np
import random
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset
from scipy.ndimage import rotate
import torch
from PIL import Image
import h5py
from utils import is_file_older, get_pix_size, rotate_tsdf, get_pix_from_pos, center_crop, rand_from_range, rand_from_low_high
from tqdm import tqdm
import pybullet as p
import matplotlib.pyplot as plt
from copy import deepcopy

class SceneDataset(Dataset):
    def __init__(self, dataset_root="dataset/kit_dataset"):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.dataset_root.mkdir(parents=True, exist_ok=True)

        self.datapoint_paths = [
            datapoint_path
            for datapoint_path in self.dataset_root.iterdir()
            if datapoint_path.is_dir() and not datapoint_path.name.startswith(".")
        ]
        
    def __len__(self):
        return len(self.datapoint_paths)
    
    
    def print_statistics(self, prefix:str=""):
        print(f"{prefix} ScenedatasetStats: len({len(self)});")

    @staticmethod
    def extend(
        rgb, d, masks, labels, boxes,
        sc_inps, sc_targets,
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_dir/"data.hdf", "w") as hdf:
            # input
            hdf.create_dataset("rgb", data=rgb)
            hdf.create_dataset("d", data=d)
            # target
            # - instance masks
            hdf.create_dataset("masks", data=masks)
            hdf.create_dataset("boxes", data=boxes)
            hdf.create_dataset("labels", data=labels)
            hdf.create_dataset("sc_inps", data=sc_inps)
            hdf.create_dataset("sc_targets", data=sc_targets)

class SceneDatasetMaskRCNN(SceneDataset):
    def __init__(
        self,
        dataset_root: Path = Path("dataset/kits/"),
        use_depth: bool = False,
        transforms=None,
    ):
        super().__init__(dataset_root)
        self.use_depth = use_depth
        self.transforms = transforms

    def __getitem__(self, index):
        datapoint_path = self.datapoint_paths[index]
        with h5py.File(datapoint_path/"data.hdf", "r") as hdf:
            target = dict()
            boxes = torch.tensor(hdf.get("boxes"), dtype=torch.float32)
            target["boxes"] = boxes
            target["labels"] = torch.tensor(hdf.get("labels"), dtype=torch.int64)
            target["masks"] = torch.tensor(hdf.get("masks"))
            target["image_id"] = torch.tensor([index]) 
            target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

            def apply_transform(img, target):
                if self.transforms is not None:
                    img, target = self.transforms(img, target)
                return img, target

            rgb, d = None, None
            if not self.use_depth:
                rgb = np.array(hdf.get("rgb"))
                return apply_transform(rgb, target)
            else:
                d = np.array(hdf.get("d"))
                return apply_transform(d, target)

class SceneDatasetShapeCompletionSnap():
    def __init__(self, dataset_root, vol_type):
        print("Loading SC dataset from path: ", dataset_root)
        if isinstance(dataset_root, str):
            dataset_root = Path(dataset_root)
        self.data_paths = list(dataset_root.glob("**/data.hdf"))
        self.vol_type = vol_type
        
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        hdf_path = self.data_paths[index]
        with h5py.File(str(hdf_path), "r") as hdf:
            if self.vol_type == 'object':
                inp = np.array(hdf.get("p0_vol_raw"))
                tar = np.array(hdf.get("p0_vol"))
                pred = np.array(hdf.get("p0_vol_sc"))
            elif self.vol_type == 'kit':
                inp = np.array(hdf.get("p1_vol_raw"))
                tar = np.array(hdf.get("p1_vol"))
                pred = np.array(hdf.get("p1_vol_sc"))
        inp = np.expand_dims(inp, axis=0)
        tar = np.expand_dims(tar, axis=0)
        pred = np.expand_dims(pred, axis=0)
        return inp, tar#, pred, int(self.data_paths[index].parent.name)

class SceneDatasetShapeCompletion(SceneDataset):
    def __init__(self, dataset_root: Path, scene_type):
        super().__init__(dataset_root)
        print("Loading SC dataset from path: ", dataset_root)
        self.scene_type = scene_type
        assert self.scene_type in ['kit', 'object'], \
            'Expected scene type to be "kit" or "object".'

        if isinstance(dataset_root, str):
            dataset_root = Path(dataset_root)
        if self.scene_type == 'object':
            self.hdf = self.gen_cache(dataset_root)

    def gen_cache(self, dataset_root):
        cache_path = dataset_root / "sc_cache.hdf" 
        if not cache_path.exists() or is_file_older(self.datapoint_paths[0] / "data.hdf", cache_path):
            with h5py.File(cache_path, "w") as hdf:
                hdf_index = 0
                for datapoint_path in tqdm(self.datapoint_paths, desc="Generating sc cache", dynamic_ncols=True):
                    with h5py.File(datapoint_path / "data.hdf", "r") as data_hdf:
                        inps = np.array(data_hdf.get("sc_inps"))
                        targets = np.array(data_hdf.get("sc_targets"))
                        for inp, target in zip(inps, targets):
                            grp = hdf.create_group(str(hdf_index))
                            hdf_index += 1
                            grp.create_dataset("inp", data=inp)
                            grp.create_dataset("target", data=target)
        return h5py.File(cache_path, "r")
        
    def __len__(self):
        if self.scene_type == 'kit':
            return len(self.datapoint_paths)
        elif self.scene_type == 'object':
            return len(self.hdf.keys())

    def __getitem__(self, index):
        if self.scene_type == 'kit':
            with h5py.File(self.datapoint_paths[index] / "data.hdf", "r") as data_hdf:
                inps = np.array(data_hdf.get("sc_inps"))
                targets = np.array(data_hdf.get("sc_targets"))
            inp = np.expand_dims(np.array(inps[0]), axis=0)
            target = np.expand_dims(np.array(targets[0]), axis=0)
        elif self.scene_type == 'object':
            grp = self.hdf.get(str(index))
            inp = np.expand_dims(np.array(grp['inp']), axis=0)
            target = np.expand_dims(np.array(grp['target']), axis=0)
        return inp, target

class TNDataset(Dataset):
    def __init__(self, dataset_root, view_bounds, view_bounds_obj, view_bounds_kit, pix_size, perturb_delta):
        super().__init__()
        self.dataset_root = dataset_root
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.data_paths = list(self.dataset_root.glob("**/data.hdf"))
        self.view_bounds = view_bounds
        self.view_bounds_obj = view_bounds_obj
        self.view_bounds_kit = view_bounds_kit
        self.pix_size = pix_size
        self.perturb_delta = perturb_delta
        self.max_perturb_angle = 0.48
        self.voxel_size = 0.0008928571428571428
        self.max_perturb_delta = np.array([32,32,32])

        # max_yaw_pitch = 15
        # filted_data_paths = []
        # for hdf_path in self.data_paths:
        #     with h5py.File(str(hdf_path), "r") as hdf:
        #         p1_ori = np.array(hdf.get("p1_ori"))
        #     rotate_angles = quat_to_euler(p1_ori, degrees=True)
        #     if (np.abs(rotate_angles[0:2]) < max_yaw_pitch).all():
        #         filted_data_paths.append(hdf_path)
        # self.data_paths = filted_data_paths
        size = len(self.data_paths)

        print(f'Dataset size: {size}')

    @staticmethod
    def from_cfg(cfg, dataset_split):
        dataset_root = Path('dataset/vol_match_abc/') / dataset_split
        image_size = np.array(cfg.image_size)
        view_bounds = np.array(cfg.workspace_bounds)
        view_bounds_obj = np.array(cfg.workspace_bounds_obj)
        view_bounds_kit = np.array(cfg.workspace_bounds_kit)
        pix_size = get_pix_size(view_bounds, image_size[0])
        perturb_delta = cfg.perturb_delta
        return TNDataset(dataset_root, view_bounds, view_bounds_obj, view_bounds_kit, pix_size, perturb_delta)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_hdf = self.data_paths[index]
        with h5py.File(data_hdf, "r") as hdf:
            cmap = np.array(hdf.get("cmap"))
            hmap = np.array(hdf.get("hmap"))
            half_width = cmap.shape[1]//2
            cmap_obj = cmap[:,:half_width,:]
            hmap_obj = hmap[:,:half_width]
            cmap_kit = cmap[:,half_width:,:]
            hmap_kit = hmap[:,half_width:]
            pick_pos = np.array(hdf.get("pick_pos"), dtype=np.float)
            place_pos = np.array(hdf.get("place_pos"), dtype=np.float)
            place_ori = np.array(hdf.get("place_ori"))
            concav_ori = np.array(hdf.get("concav_ori"))
            symmetry = np.array(hdf.get("symmetry"))
            
        # Choose crop center with some perturbation around the p1_coords
        perturb_delta = np.array([rand_from_range(r) for r in self.max_perturb_delta])
        user_pos = place_pos - perturb_delta * self.voxel_size
        # Perturb orientation
        perturb_phi = self.max_perturb_angle * random.random()
        perturb_theta = np.pi * 2 * random.random()
        vec = np.array([0,-np.sin(perturb_phi),np.cos(perturb_phi)])
        rotm = np.array([[np.cos(perturb_theta), -np.sin(perturb_theta), 0],
                        [np.sin(perturb_theta), np.cos(perturb_theta), 0],
                        [0,0,1]])
        vec = rotm @ vec
        perturb_quat = normal_to_quat(vec)
        user_quat = multiply_quat(place_ori, perturb_quat)

        return cmap_obj, hmap_obj, cmap_kit, hmap_kit, pick_pos, place_pos, place_ori, concav_ori, symmetry, user_pos, user_quat
    
    @staticmethod         
    def visualize_tn(output_dir, num, view_bounds_obj, view_bounds_kit, pix_size, cmap_obj, cmap_kit, syms, gt, pred=None, diffs=None):
        
        dump_paths = dict()
        pad_size = 54
        cmap_obj = np.pad(cmap_obj, [[pad_size,pad_size], [pad_size,pad_size], [0,0]])
        cmap_kit = np.pad(cmap_kit, [[pad_size,pad_size], [pad_size,pad_size], [0,0]])
        
        def get_color_img(img: np.array):
            color_img = img.astype(np.uint8)
            return Image.fromarray(color_img)
        def get_overlay(pick_pos, place_pos, place_ori, prefix=''):
            p0_pix = get_pix_from_pos(pick_pos, view_bounds_obj, pix_size)
            p1_pix = get_pix_from_pos(place_pos, view_bounds_kit, pix_size)
            z = place_pos[2]
            roll, pitch, yaw = quat_to_euler(place_ori, degrees=True) # degrees
            p0_pix_pad = p0_pix + pad_size
            p1_pix_pad = p1_pix + pad_size
            px, py = p0_pix_pad
            qx, qy = p1_pix_pad

            cmap_obj_copy = cmap_obj.copy()
            cmap_obj_copy[px-10:px+10, py-10:py+10, :] = 255
            get_color_img(cmap_obj_copy).save(output_dir / f'{num}_{prefix}_pick.jpg')
            dump_paths[f'{prefix}_pick'] = output_dir / f'{num}_{prefix}_pick.jpg'

            cmap_kit_copy = cmap_kit.copy()
            cmap_kit_copy[qx-10:qx+10, qy-10:qy+10, 0] = 255
            cmap_kit_copy[qx-10:qx+10, qy-10:qy+10, 1:] = 0
            get_color_img(cmap_kit_copy).save(output_dir / f'{num}_{prefix}_place.jpg')
            dump_paths[f'{prefix}_place'] = output_dir / f'{num}_{prefix}_place.jpg'

            p0_crop = cmap_obj[px-54:px+54, py-54:py+54, :]
            p0_crop = rotate(p0_crop, yaw, reshape=False, mode='nearest')
            cmap_kit_copy = cmap_kit.copy()
            cmap_kit_copy[qx-54:qx+54, qy-54:qy+54, :] = 0.5*cmap_kit[qx-54:qx+54, qy-54:qy+54, :] + 0.5*p0_crop
            get_color_img(cmap_kit_copy).save(output_dir / f'{num}_{prefix}_overlay.jpg')
            dump_paths[f'{prefix}_overlay'] = output_dir / f'{num}_{prefix}_overlay.jpg'

            rot_gt_str = np.array2string(np.array([roll, pitch, yaw]), 
                                        precision=2, separator='   ', suppress_small=True)
            dump_paths[f'{prefix}_ori_z'] = f'rot: {rot_gt_str}, z: {z:.2f}'
        
        pick_pos_gt, place_pos_gt, place_ori_gt = gt
        get_overlay(pick_pos_gt, place_pos_gt, place_ori_gt, 'gt')
        dump_paths['symmetry'] = f'x: {syms[0]}, y: {syms[1]}, z: {syms[2]}'
        if pred is not None:
            place_pos_pred, place_ori_pred = pred
            get_overlay(pick_pos_gt, place_pos_pred, place_ori_pred, 'pred')
            dump_paths[f'diff'] = f'pos_diff: {diffs[0]:.3f} mm, rot_diff: {diffs[1]:.3f} deg'

        return dump_paths
                
class VolMatchDataset(Dataset):
    def __init__(self, dataset_root:Path, p0_vol_shape: np.ndarray, p1_vol_shape: np.ndarray, 
                 max_perturb_delta: np.ndarray, max_perturb_angle, size: int, vol_type: str, 
                 no_user_input: bool, max_yaw_pitch: float, min_d: int, min_angle: float, use_gt_ori):
        super().__init__()
        self.dataset_root = dataset_root 
        self.dataset_root.mkdir(exist_ok=True, parents=True)
        self.p0_vol_shape = p0_vol_shape
        self.p1_vol_shape = p1_vol_shape
        self.max_perturb_delta = max_perturb_delta
        self.max_perturb_angle = max_perturb_angle
        self.data_paths = list(self.dataset_root.glob("**/data.hdf"))
        size = min(size, len(self.data_paths))
        self.data_paths = self.data_paths[:size]
        self.vol_type = vol_type
        self.no_user_input = no_user_input
        self.max_yaw_pitch = max_yaw_pitch
        self.min_d = min_d
        self.min_angle = min_angle
        self.use_gt_ori = use_gt_ori
        if self.no_user_input and 'val' in str(self.dataset_root):
            filted_data_paths = []
            for hdf_path in self.data_paths:
                with h5py.File(str(hdf_path), "r") as hdf:
                    p1_ori = np.array(hdf.get("p1_ori"))
                rotate_angles = quat_to_euler(p1_ori, degrees=True)
                if (np.abs(rotate_angles[0:2]) < max_yaw_pitch).all():
                    filted_data_paths.append(hdf_path)
            self.data_paths = filted_data_paths
            size = len(self.data_paths)
        print(f"Using VolMatchDataset from {self.dataset_root}, size={size}, vol_type: {self.vol_type}, no_user_input: {no_user_input}")
    
    @staticmethod
    def from_cfg(cfg: DictConfig, dataset_path, vol_type=None):
        vm_cfg = cfg.vol_match_6DoF
        max_perturb_delta = np.array(vm_cfg.max_perturb_delta)
        p0_vol_shape = np.array(vm_cfg.p0_vol_shape)
        p1_vol_shape = np.array(vm_cfg.p1_vol_shape)
        if vol_type is None:
            vol_type = vm_cfg.vol_type
        max_perturb_angle = vm_cfg.max_perturb_angle
        size = int(vm_cfg.dataset_size)
        no_user_input = bool(vm_cfg.no_user_input)
        max_yaw_pitch = float(vm_cfg.max_yaw_pitch)
        min_d = float(vm_cfg.min_d)
        min_angle = float(vm_cfg.min_angle)
        use_gt_ori = vm_cfg.use_gt_ori
        return VolMatchDataset(dataset_path, p0_vol_shape, p1_vol_shape, max_perturb_delta, max_perturb_angle, size, 
                                vol_type, no_user_input, max_yaw_pitch, min_d, min_angle, use_gt_ori)
    
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        hdf_path = self.data_paths[index]
        with h5py.File(str(hdf_path), "r") as hdf:
            if self.vol_type == 'oracle':
                p0_vol = np.array(hdf.get("p0_vol"))
                p1_vol = np.array(hdf.get("p1_vol"))
            elif self.vol_type == 'sc':
                p0_vol = np.array(hdf.get("p0_vol_sc"))
                p1_vol = np.array(hdf.get("p1_vol_sc"))
            elif self.vol_type == 'raw':
                p0_vol = np.array(hdf.get("p0_vol_raw"))
                p1_vol = np.array(hdf.get("p1_vol_raw"))
            p1_coords = np.ceil(hdf.get("p1_coords")).astype(np.int)
            p1_ori = np.array(hdf.get("p1_ori"))
            concav_ori = np.array(hdf.get("concav_ori"))
            symmetry = np.array(hdf.get("symmetry"))
        if self.no_user_input:
            p1_coords_perturbed = np.array(p1_vol.shape)//2
            if 'val' in str(self.dataset_root):
                p1_ori_final = p1_ori
                p0_vol_rotate = p0_vol
            else:
                # perturb gt position
                p1_coords += np.array([rand_from_range(5) for _ in range(3)])
                # Perturb orientation
                while True:
                    perturb_quat = uniform_sample_quaternion()
                    perturb_angles = quat_to_euler(perturb_quat, degrees=False)
                    if (perturb_angles[:2] * 180/np.pi < self.max_yaw_pitch).all():
                        break
                rotate_angles = quat_to_euler(p1_ori, degrees=False)
                p0_vol_rotate = rotate_tsdf(p0_vol, rotate_angles, degrees=False)
                p0_vol_rotate = rotate_tsdf(p0_vol_rotate, perturb_angles,  degrees=False)
                p1_ori_final = invert_quat(perturb_quat)
        else:
            # Perturb position
            perturb_delta = np.array([rand_from_range(r) for r in self.max_perturb_delta])
            # perturb_delta = np.array([rand_from_low_high(self.min_d, self.min_d+3) for _ in range(3)])
            p1_coords_perturbed = p1_coords - perturb_delta
            for i in range(3):
                p1_coords_perturbed[i] = np.clip(p1_coords_perturbed[i], 0, p1_vol.shape[i]-1)
            rotate_angles = quat_to_euler(p1_ori, degrees=False)
            p0_vol_rotate = rotate_tsdf(p0_vol, rotate_angles, degrees=False)
            if not self.use_gt_ori:
                # Perturb orientation
                perturb_phi = self.max_perturb_angle * random.random()
                # perturb_phi = random.random() * 0.06 + self.min_angle
                perturb_theta = np.pi * 2 * random.random()
                vec = np.array([0,-np.sin(perturb_phi),np.cos(perturb_phi)])
                rotm = np.array([[np.cos(perturb_theta), -np.sin(perturb_theta), 0],
                                [np.sin(perturb_theta), np.cos(perturb_theta), 0],
                                [0,0,1]])
                vec = rotm @ vec
                perturb_quat = normal_to_quat(vec)
                perturb_angles = quat_to_euler(perturb_quat, degrees=False)
                p0_vol_rotate = rotate_tsdf(p0_vol_rotate, perturb_angles,  degrees=False)
                p1_ori_final = invert_quat(perturb_quat)
            else:
                p1_ori_final = np.array([0,0,0,1])

        sample = {
            "p0_vol": p0_vol_rotate,
            "p1_vol": p1_vol, 
            "p1_coords": p1_coords,
            "p1_coords_user": p1_coords_perturbed,
            "p1_ori": p1_ori_final,
            "concav_ori": concav_ori,
            "symmetry": symmetry
        }
        return sample
    
    @staticmethod         
    def visualize_6dof(output_dir, prefix, voxel_size, kit_shape, symmetry,
                    p0_vol, p1_vol, p1_coords, p1_coords_user, p1_ori,
                    p1_coords_pred=None, p1_ori_pred=None, p_diff=None, ori_diff=None, 
                    gui: bool = False):
        """
            Passing gui=True will show the pybullet instead of dumping the gifs
        """
        if torch.is_tensor(p1_coords):
            p1_coords = p1_coords.cpu().numpy()
        if torch.is_tensor(p1_coords_user):
            p1_coords_user = p1_coords_user.cpu().numpy()
        if torch.is_tensor(p1_ori):
            p1_ori = p1_ori.cpu().numpy()
        if torch.is_tensor(p0_vol):
            p0_vol = p0_vol.cpu().numpy()
        if torch.is_tensor(p1_vol):
            p1_vol = p1_vol.cpu().numpy()

        # crop kit volumes around user provided centers
        p1_vol = center_crop(p1_vol, p1_coords_user, kit_shape, tensor=False)
        # get gt coordinates within cropped shape
        p1_coords = p1_coords - p1_coords_user + np.array(kit_shape)//2

        dump_paths = dict()
        p0_mesh_path = output_dir / f"{prefix}_p0_vol.obj"
        if not TSDFHelper.to_mesh(p0_vol, p0_mesh_path, voxel_size):
            print(f"Error while dumping volume at {p0_mesh_path}")
            return dump_paths

        p1_mesh_path = output_dir / f"{prefix}_p1_vol.obj"
        if not TSDFHelper.to_mesh(p1_vol, p1_mesh_path, voxel_size):
            print(f"Error while dumping volume at {p1_mesh_path}")
            return dump_paths
        
        bb_min = -0.08 * np.ones((3,))
        bb_max = 0.08 * np.ones((3,))

        bb_min = -0.08 * np.ones((3,))
        bb_max = 0.08 * np.ones((3,))
        vis_env = MeshRendererEnv(gui=gui)

        # data visualization
        vis_env.load_mesh(p1_mesh_path, rgba=np.array([1, 0, 0, 0.5]))
        vis_env.load_mesh(p0_mesh_path, [0, 0, 0], [0,0,0,1], rgba=np.array([0, 1, 0, 0.5]))
        if not gui:
            dump_paths["data_vis"] = vis_env.render(output_dir / f"{prefix}_data_vis.gif", bb_min, bb_max)
        else:
            vis_env.step_simulation(1e3, sleep=True)  
        vis_env.reset()

        # gt visulization
        vis_env.load_mesh(p1_mesh_path, rgba=np.array([1, 0, 0, 0.5]))
        p0_pos_gt = (p1_coords - np.array(p1_vol.shape) / 2) * voxel_size
        vis_env.load_mesh(p0_mesh_path, p0_pos_gt, p1_ori, rgba=np.array([0, 1, 0, 0.5]))
        
        if not gui:
            dump_paths["gt_vis"] = vis_env.render(output_dir / f"{prefix}_gt_vis.gif", bb_min, bb_max)
        else:
            vis_env.step_simulation(1e3, sleep=True)
        vis_env.reset()

        pos_gt = p1_coords - np.array(p1_vol.shape) / 2
        pos_gt_str = np.array2string(pos_gt, separator='   ', suppress_small=True)
        euler_gt = np.array(p.getEulerFromQuaternion(p1_ori)) * 180/np.pi
        rot_gt_str = np.array2string(euler_gt, precision=2, separator='   ', suppress_small=True)
        dump_paths['gt'] = [f'pos: {pos_gt_str}', f'ori: {rot_gt_str}']
        dump_paths['symmetry'] = [f'x: {symmetry[0]}, y: {symmetry[1]}, z: {symmetry[2]}']
        # pred visulization
        if p1_coords_pred is not None and p1_ori_pred is not None:
            # get pred coordinates within cropped shape
            p1_coords_pred = p1_coords_pred - p1_coords_user + np.array(kit_shape)//2
            vis_env.load_mesh(p1_mesh_path, rgba=np.array([1, 0, 0, 0.5]))
            p1_pos_pred = (p1_coords_pred - np.ceil(np.array(p1_vol.shape) / 2).astype(np.int)) * voxel_size
            vis_env.load_mesh(p0_mesh_path, p1_pos_pred, p1_ori_pred, rgba=np.array([0, 1, 0, 0.5]))
            dump_paths["pred_vis"] = vis_env.render(output_dir / f"{prefix}_pred_vis.gif", bb_min, bb_max)
            vis_env.reset()

            euler_pred = np.array(p.getEulerFromQuaternion(p1_ori_pred)) * 180/np.pi
            dump_paths['diff'] = [f'pos: {p_diff:.3f}', f'ori: {ori_diff:.3f}']
        
        return dump_paths

class ResultDataset(Dataset):
    def __init__(self, dataset_root:Path, p0_vol_shape: np.ndarray, p1_vol_shape: np.ndarray, 
                 max_perturb_delta: np.ndarray, max_perturb_angle, size: int, voxel_size: float,
                 realworld: bool, no_user:bool):
        super().__init__()
        self.dataset_root = dataset_root 
        self.dataset_root.mkdir(exist_ok=True, parents=True)
        self.p0_vol_shape = p0_vol_shape
        self.p1_vol_shape = p1_vol_shape
        self.max_perturb_delta = max_perturb_delta
        self.max_perturb_angle = max_perturb_angle
        self.data_paths = list(self.dataset_root.glob("**/data.hdf"))
        size = min(size, len(self.data_paths))
        self.data_paths = self.data_paths[:size]
        self.voxel_size = voxel_size
        self.realworld = realworld
        self.no_user = no_user
        if self.no_user and not self.realworld:
            filted_data_paths = []
            for hdf_path in self.data_paths:
                with h5py.File(str(hdf_path), "r") as hdf:
                    all_p1_ori = np.array(hdf.get("all_p1_ori"))
                good_sample = True
                for p1_ori in all_p1_ori:
                    rotate_angles = quat_to_euler(p1_ori, degrees=True)
                    if (np.abs(rotate_angles[0:2]) > 20).any():
                        good_sample = False
                        break
                if good_sample:
                    filted_data_paths.append(hdf_path)
            self.data_paths = filted_data_paths
            size = len(self.data_paths)
        print(f"Using ResultDataset from {self.dataset_root}, size={size}")
    
    @staticmethod
    def from_cfg(cfg: DictConfig, dataset_path, realworld):
        vm_cfg = cfg.vol_match_6DoF
        max_perturb_delta = np.array(vm_cfg.max_perturb_delta)
        p0_vol_shape = np.array(vm_cfg.p0_vol_shape)
        p1_vol_shape = np.array(vm_cfg.p1_vol_shape)
        max_perturb_angle = vm_cfg.max_perturb_angle
        size = int(vm_cfg.dataset_size)
        voxel_size = cfg.env.voxel_size
        no_user = vm_cfg.no_user_input
        return ResultDataset(
            dataset_path, p0_vol_shape, p1_vol_shape, 
            max_perturb_delta, max_perturb_angle, size, voxel_size,
            realworld, no_user)

    def simulate_user_sample(
        self, p0_vol, p0_vol_sc, p0_vol_raw, p1_vol_shape, 
        p1_coords, p1_ori, place_pos, place_ori):
        # Perturb position
        perturb_delta = np.array([rand_from_range(r) for r in self.max_perturb_delta])
        p1_coords_perturbed = p1_coords - perturb_delta
        for i in range(3):
            p1_coords_perturbed[i] = np.clip(p1_coords_perturbed[i], 0, p1_vol_shape[i]-1)
        rotate_angles = quat_to_euler(p1_ori, degrees=False)
        p0_vol_rotate = rotate_tsdf(p0_vol, rotate_angles, degrees=False)
        p0_vol_sc_rotate = rotate_tsdf(p0_vol_sc, rotate_angles, degrees=False)
        p0_vol_raw_rotate = rotate_tsdf(p0_vol_raw, rotate_angles, degrees=False)

        # Perturb orientation
        perturb_phi = self.max_perturb_angle * random.random()
        perturb_theta = np.pi * 2 * random.random()
        vec = np.array([0,-np.sin(perturb_phi),np.cos(perturb_phi)])
        rotm = np.array([[np.cos(perturb_theta), -np.sin(perturb_theta), 0],
                        [np.sin(perturb_theta), np.cos(perturb_theta), 0],
                        [0,0,1]])
        vec = rotm @ vec
        perturb_quat = normal_to_quat(vec)
        perturb_angles = quat_to_euler(perturb_quat, degrees=False)
        p0_vol_rotate = rotate_tsdf(p0_vol_rotate, perturb_angles,  degrees=False)
        p0_vol_sc_rotate = rotate_tsdf(p0_vol_sc_rotate, perturb_angles,  degrees=False)
        p0_vol_raw_rotate = rotate_tsdf(p0_vol_raw_rotate, perturb_angles,  degrees=False)
        p1_ori_final = invert_quat(perturb_quat)

        user_pos = place_pos - perturb_delta * self.voxel_size
        user_quat = multiply_quat(place_ori, perturb_quat)

        return p0_vol_rotate, p0_vol_sc_rotate, p0_vol_raw_rotate, p1_coords_perturbed, p1_ori_final, user_pos, user_quat

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        hdf_path = self.data_paths[index]
        with h5py.File(str(hdf_path), "r") as hdf:
            rgb = np.array(hdf.get("rgb")).astype(int)
            d = np.array(hdf.get("d")).astype(float)
            
            all_pos = np.array(hdf.get("all_pos"))
            kit_pos = np.array(hdf.get("kit_pos"))

            p1_vol = np.array(hdf.get("p1_vol"))
            p1_vol_sc = np.array(hdf.get("p1_vol_sc"))
            p1_vol_raw = np.array(hdf.get("p1_vol_raw"))
            p1_mask = np.array(hdf.get("p1_mask"))

            all_p0_vol = np.array(hdf.get("all_p0_vol"))
            all_p0_vol_raw = np.array(hdf.get("all_p0_vol_raw"))
            all_p0_vol_sc = np.array(hdf.get("all_p0_vol_sc"))
            
            all_p1_coords = np.ceil(hdf.get("all_p1_coords")).astype(int)
            all_p1_ori = np.array(hdf.get("all_p1_ori"))

            all_pred_masks = np.array(hdf.get("all_pred_masks"))
        
        half_kit_shape = np.array(p1_vol.shape)//2
        # tn
        tn_d = deepcopy(d)
        # if self.realworld:
        height_dis = -0.065
        tn_d[:2] -= height_dis
        intrinsics = get_intrinsics(realworld=True)
        xyz = get_pointcloud(tn_d, intrinsics)
        tn_image_size = (640, 1280)
        pix_size = get_pix_size(get_tn_bounds()[0], tn_image_size[0])
        transform = np.load('real_world/dataset/camera_pose.npy')
        xyz = transform_pointcloud(xyz, transform)
        hmap, cmap = get_heightmap(xyz, rgb, get_tn_bounds()[0], pix_size)
        half_width = cmap.shape[1]//2
        cmap_obj, hmap_obj = cmap[:,half_width:,:], hmap[:,half_width:]
        cmap_kit, hmap_kit = cmap[:,:half_width,:], hmap[:,:half_width]
        
        num_obj = len(all_p0_vol)
        all_p0_vol_rotate, all_sample_sc, all_sample_raw, all_sample_tn = [], [], [], []
        for i in range(num_obj):
            p1_coords = all_p1_coords[i]
            p1_ori = all_p1_ori[i]
            p0_vol_raw = all_p0_vol_raw[i]
            p0_vol_sc = all_p0_vol_sc[i]
            p0_vol = all_p0_vol[i]

            pick_pos = all_pos[i]
            place_pos = kit_pos + (p1_coords-half_kit_shape) * self.voxel_size
            place_ori = p1_ori
            # if self.realworld:
            place_pos[2] -= height_dis
            pick_pos[2] -= height_dis

            if self.no_user:
                p0_vol_rotate, p0_vol_sc_rotate, p0_vol_raw_rotate, p1_coords_perturbed, p1_ori_final, user_pos, user_quat = \
                    p0_vol, p0_vol_sc, p0_vol_raw, half_kit_shape, p1_ori, place_pos, place_ori
            else:
                p0_vol_rotate, p0_vol_sc_rotate, p0_vol_raw_rotate, p1_coords_perturbed, p1_ori_final, user_pos, user_quat = \
                    self.simulate_user_sample(p0_vol, p0_vol_sc, p0_vol_raw, p1_vol.shape, p1_coords, p1_ori, place_pos, place_ori)
            all_p0_vol_rotate.append(p0_vol_rotate)

            concav_ori, symmetry = np.array([0,0,0,1]), np.array([-1,-1,-1])
            sample_sc = {
                "p0_vol": p0_vol_sc_rotate,
                "p1_vol": p1_vol_sc,
                "p1_coords": p1_coords,
                "p1_coords_user": p1_coords_perturbed,
                "p1_ori": p1_ori_final,
                "concav_ori": concav_ori,
                "symmetry": symmetry
            }
            sample_raw = deepcopy(sample_sc)
            sample_raw['p0_vol'], sample_raw['p1_vol'] = p0_vol_raw_rotate, p1_vol_raw
            sample_tn = (
                cmap_obj, hmap_obj, cmap_kit, hmap_kit, 
                pick_pos, place_pos, place_ori, 
                concav_ori, symmetry, user_pos, user_quat
            )
            all_sample_sc.append(sample_sc)
            all_sample_raw.append(sample_raw)
            all_sample_tn.append(sample_tn)
        return rgb, d, all_pos, all_pred_masks, p1_mask, kit_pos, p1_vol, \
            all_p0_vol_rotate, all_sample_sc, all_sample_raw, all_sample_tn

class Depth2OrientDataset(Dataset):
    def __init__(self, dataset_root, augment=True):
        super().__init__()
        self.data_paths = list(dataset_root.glob("**/data.hdf"))
        self.augment = augment

        max_yaw_pitch = 15
        filted_data_paths = []
        for hdf_path in self.data_paths:
            with h5py.File(str(hdf_path), "r") as hdf:
                p1_ori = np.array(hdf.get("p1_ori"))
            rotate_angles = quat_to_euler(p1_ori, degrees=True)
            if (np.abs(rotate_angles[0:2]) < max_yaw_pitch).all():
                filted_data_paths.append(hdf_path)
        self.data_paths = filted_data_paths
        size = len(self.data_paths)

        print('Dataset size: ', size)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        hdf_path = self.data_paths[index]
        with h5py.File(str(hdf_path), "r") as hdf:
            part_img = np.array(hdf.get("part_img"))
            kit_img = np.array(hdf.get("kit_img"))
            pc = np.array(hdf.get("pc"))
            ori = np.array(hdf.get("ori"))
            concav_ori = np.array(hdf.get("concav_ori"))
            symmetry = np.array(hdf.get("symmetry"))
        part_img = self.standardize(part_img)
        kit_img = self.standardize(kit_img)
        if self.augment:
            part_img = self.aug(part_img)
            kit_img = self.aug(kit_img)
        return part_img, kit_img, pc, ori, concav_ori, symmetry

    @staticmethod
    def standardize(img):
        depth_mean = 0.25
        depth_std = 0.025
        img = (img - depth_mean) / depth_std
        return img
    
    @staticmethod
    def random_cut(img):
        h, w = img.shape
        a = np.zeros((h,w))
        l = int(w * 0.05)
        s = int(np.random.rand()*h)
        a[s:s+l, :] = 1
        a = rotate(a, np.random.rand()*180, reshape=False).astype(np.int)
        img[a>0] = 0
        return img

    @staticmethod
    def random_zero_out(img):
        prob = np.random.rand(*img.shape)
        img[prob<0.02] = 0
        return img

    @staticmethod
    def random_crop(img):
        h, w = img.shape
        l = max(h,w)
        m = int(l*0.1)
        img = np.pad(img, ((m, m),(m, m)), constant_values=1)
        hp, wp = (np.random.rand(2) * 2 * m - m).astype(np.int)
        img = img[m+hp:m+hp+h, m+wp:m+wp+w]
        return img

    def aug(self, img):
        img = self.random_crop(img)
        img = self.random_zero_out(img)
        img = self.random_cut(img)
        return img

    @staticmethod         
    def visualize_depth2orient(output_dir, part_img, kit_img, pc, ori, concav_ori, symmetry, ori_pred=None):
        output_dir.mkdir(exist_ok=True)
        if torch.is_tensor(part_img):
            part_img = part_img.cpu().numpy()
        if torch.is_tensor(kit_img):
            kit_img = kit_img.cpu().numpy()
        if torch.is_tensor(ori):
            ori = ori.cpu().numpy()
        if torch.is_tensor(concav_ori):
            concav_ori = concav_ori.cpu().numpy()
        if torch.is_tensor(symmetry):
            symmetry = symmetry.cpu().numpy()
        def convert_depth_img_range(d, dst_range = (0,1), cur_range = None, max_cutoff=1):
            depth = d.copy()
            depth[depth>max_cutoff]=max_cutoff
            if cur_range is None:
                cur_min, cur_max = np.min(depth), np.max(depth)
            else:
                cur_min, cur_max = cur_range
            min_val, max_val = dst_range
            scaled = (depth-cur_min)/(cur_max-cur_min)
            moved_to_range = scaled * (max_val-min_val) + min_val
            return moved_to_range
        def get_depth_img(img: np.array, cmap='hot', resize=False):
            depth_img = convert_depth_img_range(img)
            if cmap is not None:
                cm = plt.get_cmap(cmap)
                colored_image = cm(depth_img)
                depth_img = colored_image[:,:,:3]
            depth_img = (depth_img*255).astype(np.uint8)
            pil_img = Image.fromarray(depth_img)
            if resize:
                pil_img = pil_img.resize((256,256))
            return pil_img
    
        dump_paths = dict()
        part_img_PIL = get_depth_img(part_img)
        part_img_PIL.save(output_dir / 'part_img.jpg')
        dump_paths['part_img'] = output_dir / 'part_img.jpg'

        kit_img_PIL = get_depth_img(kit_img)
        kit_img_PIL.save(output_dir / 'kit_img.jpg')
        dump_paths['kit_img'] = output_dir / 'kit_img.jpg'
        
        euler = quat_to_euler(ori, degrees=True)
        part_img_rot = rotate(part_img_PIL, euler[2], reshape=False, mode='nearest')
        Image.fromarray(part_img_rot).save(output_dir / 'part_img_rot.jpg')
        dump_paths['part_img_rot'] = output_dir / 'part_img_rot.jpg'

        overlay = 0.5*np.array(kit_img_PIL) + 0.5*np.array(part_img_rot)
        Image.fromarray(overlay.astype(np.uint8)).save(output_dir / 'overlay.jpg')
        dump_paths['overlay'] = output_dir / 'overlay.jpg'

        rot_gt_str = np.array2string(np.array(euler), precision=2, separator='   ', suppress_small=True)
        dump_paths['ori'] = rot_gt_str

        symmetry_str = np.array2string(np.array(symmetry), separator='   ', suppress_small=True)
        dump_paths['symmetry'] = symmetry_str

        if ori_pred is not None:
            if torch.is_tensor(ori_pred):
                ori_pred = ori_pred.cpu().numpy()
            euler_pred = quat_to_euler(ori_pred, degrees=True)
            part_img_rot_pred = rotate(part_img_PIL, euler_pred[2], reshape=False, mode='nearest')
            Image.fromarray(part_img_rot_pred).save(output_dir / 'part_img_rot_pred.jpg')
            dump_paths['part_img_rot_pred'] = output_dir / 'part_img_rot_pred.jpg'

            overlay_pred = 0.5*np.array(kit_img_PIL) + 0.5*np.array(part_img_rot_pred)
            Image.fromarray(overlay_pred.astype(np.uint8)).save(output_dir / 'overlay_pred.jpg')
            dump_paths['overlay_pred'] = output_dir / 'overlay_pred.jpg'

            rot_pred_str = np.array2string(np.array(euler_pred), precision=2, separator='   ', suppress_small=True)
            dump_paths['ori_pred'] = rot_pred_str

            dump_paths['ori_diff'] = f'{get_quat_diff(ori, ori_pred)*180/np.pi:.2f}'
        
        return dump_paths