# This file contains code for labeling real real world dataset
import sys
from pathlib import Path
root_path = Path(__file__).parent.parent.absolute()
# print("Adding to python path: ", root_path)
sys.path = [str(root_path)] + sys.path

import time
import os, h5py
from real_world.gen_kit_unvisible_view_mask import get_kit_unvisible_vol_indices
from real_world.dataset import REAL_DATASET
from scipy.signal import convolve2d
from scipy.ndimage import zoom
from learning.vol_match_transport import VolMatchTransport
from learning.vol_match_rotate import VolMatchRotate
from icecream import ic as print_ic
import math3d as m3d
from evaluate.evaluate_model import dump_seg_vis
from shutil import copyfile, rmtree
from environment.meshRendererEnv import dump_vol_render_gif, dump_tsdf_vis, MeshRendererEnv
from utils import ensure_vol_shape, get_device, get_masked_rgb, get_masked_d, \
    get_bounds_from_center_pt, mkdir_fresh, pad_crop_to_size, rotate_tsdf, show_overlay_image
import torch
import open3d as o3d
from utils.pointcloud import PointCloud
from PIL import Image
from environment.utils import SCENETYPE
from real_world.utils import color_mask_rgb, ensure_minus_pi_to_pi, fix_ur5_rotation, get_crops_wb, get_kit_bounds, get_kit_bounds_mask, get_obj_bounds, get_obj_masks_tilted, get_workspace_bounds, \
    get_tool_init, get_obj_masks, get_client_frame_pose, transform_mask, get_obj_masks_tilted, clip_angle, get_obj_bounds_mask
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
from learning.srg import SRG
from environment.real.ur5 import UR5_URX
import json
from data_generation import get_center_pt_from_d
from omegaconf import DictConfig
import hydra
from time import sleep
from tqdm import tqdm
from random import sample
from utils.tsdfHelper import TSDFHelper
from matplotlib import pyplot as plt
from environment.real.cameras import RealSense
import numpy as np

def get_crop_bounds(bounds, vol_shape, voxel_size):
    vol_size = vol_shape * voxel_size
    crop_bounds = np.empty((3, 2))
    bounds_center = bounds.mean(axis=1)
    crop_center = np.empty(3)
    crop_center[:2] = bounds_center[:2]
    crop_center[2] = bounds[2, 0] + vol_size[2] / 2
    crop_bounds[:, 0] = crop_center - vol_size / 2
    crop_bounds[:, 1] = crop_center + vol_size / 2
    return crop_bounds

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    SEND_OBJ_PCL = True

    dataset = REAL_DATASET(Path("real_world/dataset/"))
    camera_pose = dataset.camera_pose
    camera_depth_intr = dataset.camera_depth_intr

    bounds_ws = get_workspace_bounds()
    bounds_obj = get_obj_bounds()
    bounds_kit = get_kit_bounds()

    scene_path = mkdir_fresh(Path(cfg.perception.scene_path))
    last_mTime = 0
    client_scene_path = Path("visualizer/server/updated_scene.json")
    if client_scene_path.exists():
        client_scene_path.unlink()
    update_delta = 1e-5

    # Setup segmentation and shape completion
    obj_vol_shape = np.array(cfg.env.obj_vol_shape)
    hw = np.ceil(obj_vol_shape / 2).astype(np.int)  # half width
    voxel_size = cfg.env.voxel_size

    kit_vol_shape = np.array(cfg.env.kit_vol_shape)
    kit_crop_bounds = get_crop_bounds(bounds_kit, kit_vol_shape, voxel_size)
    obj_ws_crop_bounds = get_crop_bounds(
        bounds_obj, kit_vol_shape, voxel_size)  # kit_vol_shape is intentional
    # print_ic(bounds_kit, kit_crop_bounds)

    kit_unvisible_vol_indices = get_kit_unvisible_vol_indices()

    # Get Real (robot frame) <-> Client transformations
    # - Client pose in real frame
    client_pos__real, client_ori__real = get_client_frame_pose()
    client_ori__real = p.getQuaternionFromEuler(client_ori__real)
    # - Real (robot frame) pose in client frame
    real_pos__client, real_ori__client = p.invertTransform(
        client_pos__real, client_ori__real)
    # - Transformation from client to real
    rot_mat = np.array(p.getMatrixFromQuaternion(
        client_ori__real)).reshape((3, 3))
    real__T__client = np.eye(4)
    real__T__client[:3, :3] = rot_mat
    real__T__client[:3, 3] = client_pos__real
    # - Transformation from real to client
    client__T__real = np.linalg.inv(real__T__client)

    def transform_to_client(pos__real, ori__real):
        pos__client, ori__client = p.multiplyTransforms(
            real_pos__client, real_ori__client, pos__real, ori__real)
        return list(pos__client), list(ori__client)

    def transform_to_real(pos__client, ori__client):
        pos__real, ori__real = p.multiplyTransforms(
            client_pos__real, client_ori__real, pos__client, ori__client)
        return list(pos__real), list(ori__real)

    # Add the bounds
    zero_orientation = p.getQuaternionFromEuler([0, 0, 0])
    workspace_bounds_dict = {
        "translation": [
            transform_to_client(bounds_ws[:, 0], zero_orientation)[0],
            transform_to_client(bounds_ws[:, 1], zero_orientation)[0],
        ]
    }

    while True:
        scene_dict = dict()
        scene_dict["objects"] = list()
        scene_dict["gt_objects"] = list()
        name_mask = dict()
        name_vols = dict()
        name_obj_crop_bounds = dict()  # Used for finding the top-down grasping position
        name_transformations = dict()
        name_mesh_path = dict()

        print("Please enter the datapoint number to label:")
        dataset_i = int(input())
        print_ic(dataset_i)
        rgb, d, gt_mesh_paths, datapoint_path = dataset.__getitem__(
            dataset_i,
            use_idx_as_datapoint_folder_name=True
        )

        def update_scene_dict_mesh_path(name, mesh_path, position, color, scene_dict_key, gt_object_type = None):
            mesh_filename = f"{name}.obj"
            if mesh_path.parent != scene_path:
                copyfile(mesh_path, scene_path / mesh_filename)
            obj_dict = {
                "name": str(name),
                "path": mesh_filename,
                "position": position.tolist(),
                "orientation": [0, 0, 0, 1]
            }
            if color is not None:
                obj_dict["color"] = color.tolist()
            if gt_object_type:
                obj_dict["obj_type"] = gt_object_type
            scene_dict[scene_dict_key].append(obj_dict)

        def update_scene_dict(name, vol, vol_crop_bounds, scene_dict_key: str, center_mesh=True, obj_color=None, vs=voxel_size):
            mesh_filename = f"{name}.obj"
            mesh_path = scene_path / mesh_filename
            center = None
            if center_mesh:
                success, mesh_center = TSDFHelper.to_mesh(
                    vol, mesh_path, vs, center_mesh=center_mesh)
                if mesh_center is not None:
                    mesh_center *= vs
                    center = vol_crop_bounds.mean(axis=1) + mesh_center
            else:
                success = TSDFHelper.to_mesh(
                    vol, mesh_path, vs, center_mesh=center_mesh)
                mesh_center = np.zeros((3,))
                center = vol_crop_bounds.mean(axis=1)
            # np.save(mesh_path.parent / f"{mesh_path.name}.npy", vol)
            if success:
                update_scene_dict_mesh_path(
                    name, mesh_path, center, obj_color, scene_dict_key)
                name_mesh_path[name] = mesh_path
            return mesh_center

        def update_scene_dict_obj():
            """
            Generates segmentation and shape completion for objects
            """
            # Similar to kit, just convert everything to the volume and send a big volume over there.
            obj_mask = get_obj_bounds_mask(
                camera_pose, camera_depth_intr, d.shape)
            # show_overlay_image(obj_mask, rgb)
            obj_depth = get_masked_d(obj_mask, d)
            # fig, ax = plt.subplots(1, 3)
            # ax[0].imshow(d)
            # ax[1].imshow(obj_mask)
            # ax[2].imshow(obj_depth)
            # plt.show()
            views = [(rgb, obj_depth, camera_depth_intr, camera_pose)]
            obj_ws_vol_path = datapoint_path / 'obj_ws_vol.npy'
            if obj_ws_vol_path.exists():
                print("Using cached obj_ws_vol")
                obj_ws_vol = np.load(obj_ws_vol_path)
            else:
                obj_ws_vol = TSDFHelper(
                    views, obj_ws_crop_bounds, voxel_size, initial_value=1)
                np.save(obj_ws_vol_path, obj_ws_vol)
            # Now just dump this volume.
            name = "obj_ws"
            update_scene_dict(name, obj_ws_vol, obj_ws_crop_bounds, scene_dict_key="objects", center_mesh=False,
                              vs=voxel_size, obj_color=np.array([191, 188, 178]) / 255)
            name_vols[name] = obj_ws_vol

            # Now I also want to send the gt objects here:
            for i, gt_mesh_data in enumerate(gt_mesh_paths):
                # No. Just send the obj directly?
                if "obj" in gt_mesh_data:
                    name = f"gt_obj_{i}"
                    gt_obj_path = Path(gt_mesh_data["obj"])
                    update_scene_dict_mesh_path(name, gt_obj_path, np.zeros((3,)), np.array([
                                                133, 222, 115]) / 255, "gt_objects", gt_object_type="obj")
                    name_mesh_path[name] = gt_obj_path
                if "kit" in gt_mesh_data:
                    name = f"gt_kit_{i}"
                    gt_kit_path = Path(gt_mesh_data["kit"])
                    update_scene_dict_mesh_path(f"gt_kit_{i}", gt_kit_path, np.zeros((3,)), np.array([
                                                80, 181, 139]) / 255, "gt_objects", gt_object_type="kit")
                    name_mesh_path[name] = gt_kit_path 

        update_scene_dict_obj()

        def update_scene_dict_kit():
            kit_mask = get_kit_bounds_mask(
                camera_pose, camera_depth_intr, rgb.shape[:2])
            kit_depth = get_masked_d(kit_mask, d)
            views = [(rgb, kit_depth, camera_depth_intr, camera_pose)]
            kit_vol_path = datapoint_path / 'kit_vol.npy'
            if kit_vol_path.exists():
                kit_vol = np.load(kit_vol_path)
            else:
                kit_vol = TSDFHelper.tsdf_from_camera_data(
                    views, kit_crop_bounds, voxel_size, initial_value=1)
                np.save(kit_vol_path, kit_vol)
            kit_vol = ensure_vol_shape(kit_vol, kit_vol_shape)
            kit_vol[kit_unvisible_vol_indices] = 1
            name = "kit"
            name_vols[name] = kit_vol
            update_scene_dict(name, kit_vol, kit_crop_bounds, "objects", center_mesh=False,
                              vs=voxel_size, obj_color=np.array([191, 188, 178]) / 255)

        update_scene_dict_kit()

        # Transform to client frame
        for i, obj_dict in enumerate(scene_dict["objects"]):
            obj_pos__client, obj_ori__client = transform_to_client(
                obj_dict["position"], obj_dict["orientation"])
            scene_dict["objects"][i]["position"] = obj_pos__client
            scene_dict["objects"][i]["orientation"] = obj_ori__client

        scene_dict["bounds"] = workspace_bounds_dict
        with open(scene_path/"scene.json", "w") as f:
            json.dump(scene_dict, f)
        # print("Client scene updated with: ", scene_dict)

        pcl = PointCloud(rgb, d, camera_depth_intr)
        pcl.make_pointcloud()
        o3d_pc_full = pcl.o3d_pc
        o3d_pc_full = o3d_pc_full.transform(camera_pose)

        def save_point_cloud(bounds, pcl_path):
            xyz = np.array(o3d_pc_full.points)
            rgb = np.array(o3d_pc_full.colors)
            valid_rows = (xyz[:, 0] >= bounds[0, 0]) & (
                xyz[:, 0] <= bounds[0, 1])
            valid_rows = valid_rows & (
                (xyz[:, 1] >= bounds[1, 0]) & (xyz[:, 1] <= bounds[1, 1]))
            valid_rows = valid_rows & (
                (xyz[:, 2] >= bounds[2, 0]) & (xyz[:, 2] <= bounds[2, 1]))
            o3d_pc = o3d.geometry.PointCloud()
            o3d_pc.points = o3d.utility.Vector3dVector(xyz[valid_rows])
            o3d_pc.colors = o3d.utility.Vector3dVector(rgb[valid_rows])
            o3d_pc = o3d_pc.transform(client__T__real)
            o3d.io.write_point_cloud(str(pcl_path), o3d_pc, write_ascii=True)
            # o3d.visualization.draw_geometries([o3d_pc])

        if SEND_OBJ_PCL:
            save_point_cloud(bounds_obj, scene_path /
                             f"scene_{SCENETYPE.OBJECTS.name}_pcl.ply")
        save_point_cloud(get_kit_bounds(), scene_path /
                         f"scene_{SCENETYPE.KIT.name}_pcl.ply")

        def update_scene_dict(diff_dict):
            gt_data = dict() 
            for name, val in diff_dict.items():
                gt_pos = val["upd_pos"]
                gt_ori = val["upd_ori"]
                # Save it 
                gt_data_key = val["obj_type"]
                if gt_data_key not in gt_data:
                    gt_data[gt_data_key] = list()
                gt_data[gt_data_key].append({
                    "mesh_path": str(name_mesh_path[name]),
                    "gt_pos": gt_pos,
                    "gt_ori": gt_ori
                })

            gt_save_path = datapoint_path / "gt_labels.json"
            with open(gt_save_path, "w") as label_fp:
                json.dump(gt_data, label_fp)
            print(f"labels saved at path: {gt_save_path}")

        def update_scene_from_client(initial=True):
            nonlocal last_mTime
            if not client_scene_path.exists():
                return False
            mTime = client_scene_path.stat().st_mtime
            if last_mTime is not None and last_mTime == mTime:
                return False

            with open(client_scene_path, "r") as f:
                update_scene_json = json.load(f)
            print("Updating scene")
            diff_dict = dict()

            scene_dict_proc = dict()
            for obj_dict in scene_dict["gt_objects"]:
                scene_dict_proc[obj_dict["name"]] = obj_dict

            # Need to get an index
            # - ok. in the new dict. just find the old item. compare
            # - if any difference. add it to diff dict. that's it
            for obj_dict in update_scene_json["objects"]:
                key = obj_dict["name"]
                if key not in scene_dict_proc:
                    print(f"Unknown id {key} received")
                    continue

                curr_pos, curr_ori = np.array(scene_dict_proc[key]["position"]), np.array(
                    scene_dict_proc[key]["orientation"])
                upd_pos, upd_ori = np.array(
                    obj_dict["position"]), np.array(obj_dict["orientation"])

                if (np.abs(upd_pos - curr_pos) > update_delta).any() or (np.abs(upd_ori - curr_ori) > update_delta).any():
                    print("Position changed for ", key)
                    # Transform the difference back to real world frame
                    curr_pos__real, curr_ori__real = transform_to_real(
                        curr_pos, curr_ori)
                    upd_pos__real, upd_ori__real = transform_to_real(
                        upd_pos, upd_ori)
                    diff_dict[key] = {
                        "upd_pos": upd_pos__real,
                        "upd_ori": upd_ori__real,
                        "obj_type": obj_dict["obj_type"]
                    }
                else:
                    print(f"Alignment for {key} is not proivded. Please ===> DOUBLE CHECK <===")
            update_scene_dict(diff_dict)
            last_mTime = mTime
            return True

        print("Ready: Please place the objects and the kit.")
        while not update_scene_from_client():
            sleep(1)
        print("\n\n====================================================\n")
        input("Please reset scene and press ENTER ....")


if __name__ == "__main__":
    main()


            # pbc = BulletClient(p.GUI)
            # # At this point, we have gt poses. 
            # # load the original object space volume in the pybullet
            # # Let's first load the observed real world scene
            # obj_ws_mesh_path = name_mesh_path["obj_ws"]
            # obj_ws_urdf_path = MeshRendererEnv.dump_obj_urdf(obj_ws_mesh_path, urdf_path=debug_path / "obj_ws.urdf")
            # obj_ws_id = pbc.loadURDF(str(obj_ws_urdf_path), basePosition=obj_ws_crop_bounds.mean(axis=1))
            # kit_ws_mesh_path = name_mesh_path["kit"]
            # kit_ws_urdf_path = MeshRendererEnv.dump_obj_urdf(kit_ws_mesh_path, urdf_path=debug_path / "kit_ws.urdf")
            # kit_ws_id = pbc.loadURDF(str(kit_ws_urdf_path), basePosition=kit_crop_bounds.mean(axis=1))
            # obj_ids = list()
            # for val in gt_data.values():
            #     for obj_data in val:
            #         mesh_path = Path(obj_data["mesh_path"])
            #         urdf_path = str(mesh_path.parent / f"{mesh_path.stem}.urdf")
            #         obj_ids.append(
            #             pbc.loadURDF(
            #                 urdf_path,
            #                 basePosition=obj_data["gt_pos"],
            #                 baseOrientation=obj_data["gt_ori"]
            #             )
            #         )
            #         print("URDF loaded from path: ", urdf_path, " at location", obj_data["gt_pos"])

            # from environment.camera import SimCameraBase

            # image_size = tuple(cfg.env.image_size)
            # camera_pose_tmp = np.copy(camera_pose)
            # camera_pose_tmp[:, 1:3] = -camera_pose_tmp[:, 1:3]
            # view_matrix = np.linalg.inv(camera_pose_tmp.T).flatten()

            # print_ic(camera_color_intr) 
            # focal_length = camera_color_intr[0, 0]
            # sim_camera = SimCameraBase(view_matrix=view_matrix, image_size=image_size, focal_length=focal_length) 

            # fig, ax = plt.subplots(1, 2)
            # rgb_sim, _, _ = sim_camera.get_image()
            # ax[0].imshow(rgb_sim)

            # pbc.removeBody(obj_ws_id)
            # pbc.removeBody(kit_ws_id)
            # rgb_sim1, _, _ = sim_camera.get_image()
            # ax[1].imshow(rgb_sim1)
            # ax[1].imshow(rgb, alpha=0.9)
            # plt.show()
        
            # for _ in range(int(1e5)):
            #     pbc.stepSimulation()
            #     time.sleep(1 / 240)