# Scene Reperesentation
from utils import get_device
import torch
from data_generation import gen_masked_vol, gen_vol_from_mask, process_mask
from environment.utils import TSDFHelper, SCENETYPE
from environment.meshRendererEnv import dump_vol_render_gif
from pathlib import Path
import numpy as np
from learning.seg import get_transform
from matplotlib import pyplot as plt
from PIL import Image


class SRG:
    # Scene Reperesentation Generator
    def __init__(self, perception_cfg, hw, device: torch.device = None):
        self.hw = hw
        # self.device = get_device() if device is None else device
        self.device = torch.device("cpu")
        # #print("=======>FIXME<======= using device_cpu for kit shape completion")
        self.scene_path = Path(perception_cfg.scene_path)
        self.scene_path.mkdir(parents=True, exist_ok=True)
        # Load the mask rcnn model
        self.seg_use_gt = perception_cfg.seg.use_gt
        if self.seg_use_gt:
            print("Using ground truth segmentation")
        else:
            self.seg_model = torch.load(perception_cfg.seg.path, map_location=self.device)
            self.use_depth = perception_cfg.seg.use_depth
            normalize_depth = perception_cfg.seg.normalize_depth
            self.seg_transform = get_transform(False, self.use_depth, normalize_depth)
        self.seg_score_threshold = perception_cfg.seg.mask_score_threshold
        self.seg_threshold = perception_cfg.seg.mask_threshold
        # Load the shape completion model
        self.sc_use_gt = perception_cfg.sc.use_gt
        if self.sc_use_gt:
            print("Using ground truth shape completion")
        else:
            self.sc_model = torch.load(perception_cfg.sc.path, map_location=self.device)
            self.sc_model.eval()
        self.kit_path = None
        self.kit_vol = None
    
    def __del__(self):
        if not self.seg_use_gt:
            del self.seg_model
        if not self.sc_use_gt:
            del self.sc_model

    def get_seg(self, rgb, d):
        # Segmentation 
        # - input: rgb
        # - output: masks, labels, scores
        img = d if self.use_depth else rgb
        img, _ = self.seg_transform(img, None)
        img = img.to(self.device, dtype=torch.float)
        # masks
        with torch.no_grad():
            seg_out = self.seg_model.forward([img])[0]
        masks = None
        for raw_mask in seg_out["masks"].cpu().numpy():
            mask = np.empty_like(raw_mask, dtype=np.uint8) 
            mask[raw_mask < self.seg_threshold] = 0
            mask[raw_mask >= self.seg_threshold] = 1
            if masks is None:
                masks = mask
            else:
                masks = np.concatenate((masks, mask), axis=0)

        return masks, seg_out["labels"].cpu().numpy(), seg_out["scores"].cpu().numpy()

    def reconstruct_scene(self, scene_type, start_index, env, obj_body_id_labels, is_first=False):
        vol, rgb, d, mask = env.get_scene_volume(scene_type, return_first_image=True)
        Image.fromarray(rgb).save(self.scene_path / f"scene_{scene_type.name}_rgb.png")
        if self.seg_use_gt:
            masks, labels, _, valid_indices, _ = process_mask(
                mask, obj_body_id_labels)
            masks = masks[valid_indices]
            labels = labels[valid_indices]
            scores = np.ones_like(labels)
        else:
            masks, labels, scores = self.get_seg(rgb, d)

        scene_dict = dict()
        scene_dict["objects"] = list()
        camera = env.get_camera(scene_type)
        vb = env.get_view_bounds(scene_type)
        name_mask_index = dict() 
        sc_vols = dict()
        for i in range(len(masks)):
            # Visualize prediction: RGB image. Mask.
            if labels[i] == 2:
                if scores[i] < self.seg_score_threshold:
                    continue
                
                cropped_vol_indices, center_pt = gen_vol_from_mask(
                    masks[i], d, camera.intrinsics, camera.pose_matrix, vb, env.voxel_size, self.hw, vol.shape)

                if (cropped_vol_indices == -1).any():
                    continue

                sc_inp = vol[
                    cropped_vol_indices[0, 0]:cropped_vol_indices[0, 1],
                    cropped_vol_indices[1, 0]:cropped_vol_indices[1, 1],
                    cropped_vol_indices[2, 0]:cropped_vol_indices[2, 1],
                ]
                # dump_vol_render_gif(sc_inp, self.scene_path / f"{i}_inp.obj", env.voxel_size, visualize_mesh_gif=True, visualize_tsdf_gif=True)
                sc_inp = torch.tensor(sc_inp, device=self.device).unsqueeze(dim=0).unsqueeze(dim=0)
                sc_out = self.sc_model(sc_inp).detach().squeeze().cpu().numpy()
                # dump_vol_render_gif(sc_out, self.scene_path / f"{i}_out.obj", env.voxel_size, visualize_mesh_gif=True, visualize_tsdf_gif=True)

                name = str(start_index + i)
                mesh_filename = f"{name}.obj"
                mesh_path = str(self.scene_path / mesh_filename)
                TSDFHelper.to_mesh(sc_out, mesh_path, env.voxel_size, center_mesh=True)
                sc_vols[str(name)] = sc_out
                # Hmm. I need to save these volumes somewhere
                # What's the easiest way to do this?
                # Just pass these around. Cache it. Use flags.
                # I don't want to spend hours refactoring the code now. Let's do the easiest way for now.
                # TODO: Refactor

                obj_dict = {
                    "name": str(name),
                    "path": mesh_filename,
                    "position": center_pt.tolist(),
                    "orientation": [0, 0, 0, 1]
                }
                scene_dict["objects"].append(obj_dict)
                name_mask_index[name] = i
        
        offset = (vb[:, 0] + vb[:, 1]) / 2
        # Now process kit:
        if scene_type != SCENETYPE.OBJECTS:
            if is_first:
                mask = np.ones_like(d)
                kit_vol = gen_masked_vol(mask, rgb, d, camera, vb, env.voxel_size)
                # generate the obj file
                self.kit_path = Path(self.scene_path) / "kit.obj"
                self.kit_vol = kit_vol
                TSDFHelper.to_mesh(kit_vol, self.kit_path, env.voxel_size)
            if self.kit_path is not None:
                obj_dict = {
                    "name": "kit",
                    "path": self.kit_path.name,
                    "position": offset.tolist(),
                    "orientation": [0, 0, 0, 1]
                }
                scene_dict["objects"].append(obj_dict)
                sc_vols["kit"] = self.kit_vol
        return scene_dict, rgb, d, camera, masks, name_mask_index, sc_vols

    def dump_sr(self, env, obj_body_id_labels, is_first=False):
        # The output of this whole thing should be a json file containing path
        # to object positions and 
        scene_dict1, rgb1, d1, camera1, masks1, name_mask_index1, sc_vols1 = self.reconstruct_scene(
            SCENETYPE.OBJECTS, 0, env, obj_body_id_labels, is_first)
        scene_dict2, rgb2, d2, camera2, masks2, name_mask_index2, sc_vols2 = self.reconstruct_scene(SCENETYPE.KIT, len(
            scene_dict1["objects"]), env, obj_body_id_labels, is_first)
        scene_dict = {"objects": scene_dict1["objects"] + scene_dict2["objects"]}
        masks = [masks1, masks2]
        ds = [d1, d2]
        rgbs = [rgb1, rgb2]
        cameras = [camera1, camera2]
        sc_vols = [sc_vols1, sc_vols2]
        name_mask_index = dict()
        for i, nmi in enumerate([name_mask_index1, name_mask_index2]):
            for k, v in nmi.items():
                name_mask_index[k] = (i, v)
        return scene_dict, masks, rgbs, ds, cameras, name_mask_index, sc_vols

    def visualize_predictions(self, masks, rgb, labels):
        # - get boxes
        nrows, ncols = np.ceil(len(masks) / 3).astype(np.int), 3
        fig, ax = plt.subplots(nrows, ncols, squeeze=False)
        def plt_overlayed_mask(msk, ax):
            # overlay with rgb image
            ax.imshow(rgb, cmap='gray')
            ax.imshow(msk * 255, cmap='jet', alpha=0.7) # interpolation='none'

        i, j = 0, 0
        for idx, (mask, label) in enumerate(zip(masks, labels)):
            plt_overlayed_mask(mask, ax[i][j])
            ax[i][j].set_title(f"{idx}: {'obj' if label == 2 else 'kit'} ({label})")
            if j == ncols - 1:
                j = 0
                i += 1
            else:
                j += 1
        plt.savefig(self.scene_path / f"masks_labels.png")
        plt.close(fig)
