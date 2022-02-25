# This file contains code for dumping the volumes for both the object and the kit!!
import sys
from pathlib import Path
import time
root_path = Path(__file__).parent.parent.absolute()
# print("Adding to python path: ", root_path)
sys.path = [str(root_path)] + sys.path

import numpy as np
from environment.real.cameras import RealSense
from matplotlib import pyplot as plt
from utils.tsdfHelper import TSDFHelper, extend_to_bottom
from utils.rotation import multiply_quat
from random import sample
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import hydra
from omegaconf import DictConfig
from data_generation import get_center_pt_from_d
import json
from environment.real.ur5 import UR5_URX
from learning.srg import SRG
import pybullet as p
from real_world.rw_utils import color_mask_rgb, ensure_minus_pi_to_pi, fix_ur5_rotation, get_crops_wb, get_kit_bounds, get_kit_bounds_mask, get_obj_bounds, get_obj_masks_tilted, get_workspace_bounds, \
    get_tool_init, get_obj_masks, get_client_frame_pose, transform_mask, get_obj_masks_tilted, clip_angle, get_kit_crop_bounds
from environment.utils import SCENETYPE, get_tableau_palette
from PIL import Image
from utils.pointcloud import PointCloud
from utils.tsdfHelper import get_single_biggest_cc_single
import open3d as o3d
import torch
from utils import ensure_vol_shape, get_device, get_masked_rgb, get_masked_d, \
    get_bounds_from_center_pt, mkdir_fresh, pad_crop_to_size, rotate_tsdf, show_overlay_image, center_crop
from environment.meshRendererEnv import dump_vol_render_gif, dump_tsdf_vis, MeshRendererEnv
from shutil import rmtree
from evaluate.evaluate_model import dump_seg_vis
import math3d as m3d
from icecream import ic as print_ic
from learning.vol_match_rotate import VolMatchRotate
from learning.vol_match_transport import VolMatchTransport
from real_world.gen_kit_unvisible_view_mask import get_kit_unvisible_vol_indices 
from scipy.signal import convolve2d
from real_world.dataset import REAL_DATASET
import pickle
from real_world.pyphoxi import PhoXiSensor
import cv2

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # torch.backends.cudnn.enabled = False
    RUN_DATASET = False
    # OBJECT: 
    SEND_OBJ_VOL = True
    DEBUG_SEG = SEND_OBJ_VOL and True
    DO_SC_OBJ = SEND_OBJ_VOL and True

    DEBUG_SC = DO_SC_OBJ and True
    SEND_OBJ_PCL = True
    # KIT: 
    SEND_KIT_VOL = True
    DO_SC_KIT = SEND_KIT_VOL and True
    DEBUG_SC_KIT = DO_SC_KIT and True
    SEND_KIT_PCL = True
    # Action snapping
    DO_ACTION_SNAPPING = True
    DEBUG_SNAP = DO_ACTION_SNAPPING and True
    # ROBOT: 
    RUN_ROBOT = not RUN_DATASET and True    
    SUCTION_CUP_SIZE = 0.045

    CHOOSE_BEST_YAW = False

    # Sometimes suction cup does not touch the object while picking it up.
    # We manually adjust the target grasp position by this much in z direction
    SUCTION_Z_ADJUSTMENT = -0.02
    MAX_YAW = np.pi / 3

    bin_cam = None
    dataset = None
    if not RUN_DATASET:
        # bin_cam = RealSense()
        tcp_ip = "127.0.0.1"
        tcp_port = 50200
        bin_cam = PhoXiSensor(tcp_ip, tcp_port)
        bin_cam.start()
        camera_pose = bin_cam._extr
        camera_color_intr = bin_cam._intr
    else:
        dataset = REAL_DATASET(Path("real_world/dataset/"))
        camera_pose = dataset.camera_pose
        camera_color_intr = dataset.camera_depth_intr
    #print("=======>FIXME<======= using depth intr as variable color intr")

    bounds_ws = get_workspace_bounds()
    bounds_obj = get_obj_bounds()
    bounds_kit = get_kit_bounds()

    scene_path = mkdir_fresh(Path(cfg.perception.scene_path))
    last_mTime = 0
    client_scene_path = Path("visualizer/server/updated_scene.json")
    if client_scene_path.exists():
        client_scene_path.unlink()
    update_delta = 1e-5

    # Setup robot
    tool_offset, _ = get_tool_init()
    j_acc = 0.1
    robot = None

    # Setup segmentation and shape completion
    obj_vol_shape = np.array(cfg.env.obj_vol_shape)
    hw = np.ceil(obj_vol_shape / 2).astype(int) # half width
    voxel_size = cfg.env.voxel_size
    # print_ic(obj_vol_shape, voxel_size)
    device = get_device()

    # Setup crop bounds for kit sc input
    kit_vol_shape = np.array(cfg.env.kit_vol_shape)
    kit_vol_size = kit_vol_shape * voxel_size
    # print_ic(kit_vol_size)
    kit_crop_bounds = get_kit_crop_bounds(bounds_kit, kit_vol_size)
    # print_ic(bounds_kit, kit_crop_bounds)

    if SEND_KIT_VOL:
        kit_unvisible_vol_indices = get_kit_unvisible_vol_indices()

    # setup telesnap
    vm_cfg = cfg.vol_match_6DoF 
    p1_vol_shape = np.array(vm_cfg.p1_vol_shape_gen)

    # Get Real (robot frame) <-> Client transformations
    # - Client pose in real frame
    client_pos__real, client_ori__real = get_client_frame_pose()
    client_ori__real = p.getQuaternionFromEuler(client_ori__real)
    # - Real (robot frame) pose in client frame
    real_pos__client, real_ori__client = p.invertTransform(client_pos__real, client_ori__real)
    # - Transformation from client to real
    rot_mat = np.array(p.getMatrixFromQuaternion(client_ori__real)).reshape((3,3))
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
        
    color_palette = get_tableau_palette()
    debug_path_name = cfg.perception.debug_path_name
    debug_root = Path(f"real_world/debug")
    debug_root.mkdir(exist_ok=True, parents=True)
    debug_ind = len(list(debug_root.glob('*')))
    debug_path = debug_root / f'T{debug_ind}'
    debug_path.mkdir()
    robot = None
    if RUN_ROBOT:
        robot = UR5_URX(j_vel=0.3, j_acc=0.3, tool_offset=tool_offset)
        print("Moving robot to home")
        robot.homej()
    while True:
  
        # if not RUN_DATASET:
        #     input("Please set scene and press ENTER ....")
        system_start_time = time.time()
        # Due to limited gpu resources, we unload and load models as required
        srg = None
        if SEND_OBJ_VOL:
            srg = SRG(cfg.perception, hw, device)  
        
        scene_dict = dict()
        scene_dict["objects"] = list()
        name_mask = dict()
        name_vols = dict()
        name_obj_crop_bounds = dict()  # Used for finding the top-down grasping position
        name_transformations = dict()
        all_user_pose = {}
        all_snap_pose = {}

        # Upload scene to client
        if RUN_DATASET:
            print("Please enter the datapoint number [from 0-23]:")
            debug_dir = int(input())
            rgb, d, _, _ = dataset.__getitem__(debug_dir % len(dataset), use_idx_as_datapoint_folder_name=True)
        else:
            # print_ic(debug_path)
            # rgb, d = bin_cam.get_camera_data(avg_depth=True, avg_over_n=50)
            _, gray, d = bin_cam.get_frame(True)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        plt.imshow(rgb)
        plt.savefig(debug_path / "rgb.png")
        plt.close()
        np.save(debug_path / "rgb.npy", rgb)
        
        plt.imshow(d)
        plt.savefig(debug_path / "d.png")
        plt.close()
        np.save(debug_path / "d.npy", d)

        def update_scene_dict(name, vol, vol_crop_bounds, center_mesh=True, obj_color=None, vs=voxel_size):
            mesh_filename = f"{name}.obj" 
            mesh_path = scene_path / mesh_filename
            center = None
            if center_mesh:
                success, mesh_center = TSDFHelper.to_mesh(vol, mesh_path, vs, center_mesh=center_mesh)
                if mesh_center is not None:
                    mesh_center *= vs
                    center = vol_crop_bounds.mean(axis=1) + mesh_center
            else:
                success = TSDFHelper.to_mesh(vol, mesh_path, vs, center_mesh=center_mesh)
                mesh_center = np.zeros((3,))
                center = vol_crop_bounds.mean(axis=1)
            # np.save(mesh_path.parent / f"{mesh_path.name}.npy", vol)
            if success:
                obj_dict = {
                    "name": str(name),
                    "path": mesh_filename,
                    "position": center.tolist(),
                    "orientation": [0, 0, 0, 1]
                }
                if obj_color is not None:
                    obj_dict["color"] = obj_color.tolist()
                scene_dict["objects"].append(obj_dict)
            return mesh_center

        def update_scene_dict_obj():
            """Generates segmentation and shape completion for objects
            """
            unfiltered_masks = get_obj_masks_tilted(rgb, d, camera_pose, camera_color_intr, cfg, srg)
            if unfiltered_masks is None:
                print("No object masks found. Returning")
                return
            seg_area_threshold = 1000
            masks = []
            for mask in unfiltered_masks:
                if np.sum(mask==1) > seg_area_threshold:
                    masks.append(mask)
            masks = np.array(masks)

            if DEBUG_SEG:
                # Visualize masks
                dump_seg_vis(rgb, [], [], np.ones(masks.shape[0]), masks, [], srg.seg_score_threshold, srg.seg_threshold, debug_path)
                plt.imshow(rgb)
                plt.savefig(debug_path / "rgb.png")
                plt.imshow(d)
                plt.savefig(debug_path / "d.png")

            color_indices = np.random.choice(
                np.arange(len(color_palette)), len(masks), replace=False)
            for i, mask in enumerate(masks):
                name = f"obj_{i}"
                obj_color = color_palette[color_indices[i]]
                masked_d = get_masked_d(mask, d)
                center_pt = get_center_pt_from_d(masked_d, camera_color_intr, camera_pose, bounds_ws)
                if center_pt is None:
                    print("Center pt none")
                    continue
                crop_bounds = get_bounds_from_center_pt(
                    center_pt, obj_vol_shape, voxel_size, bounds_obj)
                name_obj_crop_bounds[name] = crop_bounds
                views = [(rgb, masked_d, camera_color_intr, camera_pose)]
                sc_inp = TSDFHelper.tsdf_from_camera_data(views, crop_bounds, voxel_size)
                sc_inp = ensure_vol_shape(sc_inp, obj_vol_shape)

                if DO_SC_OBJ:
                    if DEBUG_SC:
                        np.save(debug_path / f'{i}_inp.npy', sc_inp)
                        dump_vol_render_gif(sc_inp, debug_path / f"{i}_inp.obj", voxel_size, visualize_mesh_gif=False, visualize_tsdf_gif=False)
                    sc_inp = torch.tensor(sc_inp, device=srg.device).unsqueeze(dim=0).unsqueeze(dim=0)
                    obj_vol = srg.sc_model(sc_inp).detach().squeeze().cpu().numpy()
                    obj_vol = get_single_biggest_cc_single(obj_vol)
                    obj_vol = extend_to_bottom(obj_vol)
                    if DEBUG_SC:
                        dump_vol_render_gif(obj_vol, debug_path / f"{i}_out.obj", voxel_size, visualize_mesh_gif=False, visualize_tsdf_gif=False)
                else:
                    obj_vol = sc_inp
                mesh_center = update_scene_dict(name, obj_vol, crop_bounds, center_mesh=True, obj_color=obj_color)

                name_mask[name] = mask
                name_vols[name] = obj_vol
                name_transformations[name] = mesh_center
            # pickle.dump(name_transformations, open(debug_path /'name_transformations.pkl', 'wb'))
        if SEND_OBJ_VOL:
            update_scene_dict_obj()
        del srg
        torch.cuda.empty_cache()

        # setup the kit shape completion model
        sc_kit_model = None
        if DO_SC_KIT:
            sc_kit_model = torch.load(
            Path(cfg.perception.sc_kit.path),
                map_location=device)
        def update_scene_dict_kit():
            kit_mask = get_kit_bounds_mask(camera_pose, camera_color_intr, rgb.shape[:2])
            # show_overlay_image(kit_mask, rgb)
            kit_depth = get_masked_d(kit_mask, d)
            views = [(rgb, kit_depth, camera_color_intr, camera_pose)]
            #print("=======>FIXME<======= using larger truncation margin factor for kit volume")
            kit_sc_inp = TSDFHelper.tsdf_from_camera_data(
                views, kit_crop_bounds, voxel_size)
            kit_sc_inp = ensure_vol_shape(kit_sc_inp, kit_vol_shape)
            kit_sc_inp[kit_unvisible_vol_indices] = 1
            # np.save('real_world/kit_bounds_mask.npy', kit_sc_inp)
            # dump_tsdf_vis(kit_sc_inp, debug_path / f"kit_inp_tsdf.png")
            dump_vol_render_gif(kit_sc_inp, debug_path / f"kit_inp.obj",
                                voxel_size, visualize_mesh_gif=False,
                                visualize_tsdf_gif=False)
            # Shape complete the kit volume as well:
            if DO_SC_KIT:
                # mask out the unvisible volume area   
                   
                kit_sc_inp = torch.tensor(
                    kit_sc_inp, device=device).unsqueeze(dim=0).unsqueeze(dim=0)
                kit_vol = sc_kit_model(kit_sc_inp)
                kit_vol = kit_vol.squeeze().detach().cpu().numpy()
                kit_vol = get_single_biggest_cc_single(kit_vol)
                
                if DEBUG_SC_KIT:
                    # dump_tsdf_vis(kit_vol, debug_path / f"kit_out_tsdf.png")
                    dump_vol_render_gif(kit_vol, debug_path / f"kit_out.obj",
                                        voxel_size, visualize_mesh_gif=False,
                                        visualize_tsdf_gif=False)
            else:
                kit_vol = kit_sc_inp
            name = "kit"
            name_vols[name] = kit_vol
            kit_center = update_scene_dict(name, kit_vol, kit_crop_bounds, center_mesh=False,
                              vs=voxel_size, obj_color=np.array([78, 121, 167]) / 255)
            name_transformations[name] = kit_center
        if SEND_KIT_VOL:
            update_scene_dict_kit()
        if sc_kit_model is not None:
            del sc_kit_model
            torch.cuda.empty_cache()

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

        pcl = PointCloud(rgb, d, camera_color_intr)
        pcl.make_pointcloud()
        o3d_pc_full = pcl.o3d_pc
        o3d_pc_full = o3d_pc_full.transform(camera_pose)

        def save_point_cloud(bounds, pcl_path):
            xyz = np.array(o3d_pc_full.points)
            rgb = np.array(o3d_pc_full.colors)
            valid_rows = (xyz[:, 0] >= bounds[0, 0]) & (xyz[:, 0] <= bounds[0, 1])
            valid_rows = valid_rows & ((xyz[:, 1] >= bounds[1, 0]) & (xyz[:, 1] <= bounds[1, 1]))
            valid_rows = valid_rows & ((xyz[:, 2] >= bounds[2, 0]) & (xyz[:, 2] <= bounds[2, 1]))
            o3d_pc = o3d.geometry.PointCloud()
            o3d_pc.points = o3d.utility.Vector3dVector(xyz[valid_rows])
            o3d_pc.colors = o3d.utility.Vector3dVector(rgb[valid_rows])
            o3d_pc = o3d_pc.transform(client__T__real)
            o3d.io.write_point_cloud(str(pcl_path), o3d_pc, write_ascii=True)
            # o3d.visualization.draw_geometries([o3d_pc])

        if SEND_OBJ_PCL:
            save_point_cloud(get_obj_bounds() ,scene_path / f"scene_{SCENETYPE.OBJECTS.name}_pcl.ply")
        if SEND_KIT_PCL:
            save_point_cloud(get_kit_bounds(), scene_path / f"scene_{SCENETYPE.KIT.name}_pcl.ply")

        def execute_scene_diff(diff_dict, transporter, rotator):
            # pickle.dump(diff_dict, open(debug_path / 'diff_dict.pkl', 'wb'))
            if DEBUG_SNAP:
                snap_tasks = list()
            for name, val in diff_dict.items():
                # Valid Top-down picking position:
                # - generate the volume point cloud using stored mask
                mask = name_mask[name]
                masked_rgb, masked_d = get_masked_rgb(mask, rgb), get_masked_d(mask, d)
                pc = PointCloud(masked_rgb, masked_d, camera_color_intr)  # note RGB image can be grayscale
                pc.make_pointcloud(extrinsics=camera_pose, trim=True)
                # - Find the points whose normal are vertical
                pc.compute_normals()
                ns = pc.normals
                point_cloud = pc.point_cloud

                gp_x = point_cloud[:, 0].mean()
                gp_y = point_cloud[:, 1].mean() 
                gp_z = point_cloud[:, 2].mean() + SUCTION_Z_ADJUSTMENT

                # valid_indices = ns[:, 2] > 0.99
                # if len(valid_indices) == 0:  # No vertical normal found
                #     print(f"No vertical normal found. Using position closed to the most vertical \
                #         positions {ns[:, 2].max()}")
                #     valid_indices = ns[:, 2] > ns[:, 2].max() - 0.1
                # # - Choose a graspable point from such location
                # pc_vertical = point_cloud[valid_indices][:, :3]
                # # Filter the point cloud outside view bounds
                # obj_bounds = name_obj_crop_bounds[name]
                # valid_indices = np.ones(pc_vertical.shape[0], dtype=bool)
                # valid_indices[pc_vertical[:, 0] < obj_bounds[0, 0]] = False
                # valid_indices[pc_vertical[:, 0] > obj_bounds[0, 1]] = False
                # valid_indices[pc_vertical[:, 1] < obj_bounds[1, 0]] = False
                # valid_indices[pc_vertical[:, 1] > obj_bounds[1, 1]] = False
                # valid_indices[pc_vertical[:, 2] < obj_bounds[2, 0]] = False
                # valid_indices[pc_vertical[:, 2] > obj_bounds[2, 1]] = False
                # pc_vertical = pc_vertical[valid_indices]
                # # z value: just use the mean z value of p_verticalc
                # gp_z = pc_vertical[:, 2].mean() + SUCTION_Z_ADJUSTMENT
                # # for x,y value, create binary mask by projecting pc_vertical on xy plane
                # mask_size = np.ceil(
                #     (obj_bounds[:2, 1] - obj_bounds[:2, 0]) / voxel_size).astype(int)
                # pc_mask = np.zeros((mask_size[0], mask_size[1]))
                # # For every point, project it here
                # x_indices = np.floor(
                #     (pc_vertical[:, 0] - obj_bounds[0, 0]) / voxel_size).astype(int)
                # y_indices = np.floor(
                #     (pc_vertical[:, 1] - obj_bounds[1, 0]) / voxel_size).astype(int)
                # pc_mask[x_indices, y_indices] = 1
                
                # # Now I apply an identity convolution filter of vacuum size.
                # conv_filter_size = np.ceil(SUCTION_CUP_SIZE / voxel_size).astype(int)
                # conv_filter = np.ones((conv_filter_size, conv_filter_size))
                # # Convolve the pc_mask with it.
                # pc_mask_convolved = convolve2d(pc_mask, conv_filter, mode="same")
                # # then choose the pixel with maximum value 
                # max_indices = np.unravel_index(
                #     np.argmax(pc_mask_convolved), pc_mask_convolved.shape)
                # # Visualization 
                # # fig, ax = plt.subplots(1, 2)
                # # ax[0].imshow(pc_mask)
                # # ax[1].imshow(pc_mask_convolved)
                # # marker = plt.Circle((max_indices[1], max_indices[0]), 1.5, color="red")
                # # ax[1].add_patch(marker)
                # # plt.show()
                # # Ok. Now I just need to transform this position back to world space.
                # gp_x, gp_y = np.array([max_indices[0], max_indices[1]]) * voxel_size + obj_bounds[:2, 0]
                gp = np.array([gp_x, gp_y, gp_z])
                
                final_pos, final_ori = val["upd_pos"], val["upd_ori"]
                all_user_pose[name] = (final_pos, final_ori)
                # print("Kit place pose before action snapping: ",
                #     final_pos,
                #     np.array(p.getEulerFromQuaternion(final_ori)) * 180 / np.pi
                # )
                if DO_ACTION_SNAPPING:
                    # pickle.dump(name_vols, open(debug_path /'name_vols.pkl', 'wb'))
                    p0_vol = name_vols[name]    
                    p1_vol = name_vols['kit']
                    # print_ic(p0_vol.shape, p1_vol.shape)
                    # User provided where the object mesh center (object frame of) should go in world coordinates
                    # Let's convert it to where the object vol center (volume frame vf)should go in world coordinates
                    # - pose of volume frame in object frame
                    of__P__vf = (-name_transformations[name], zero_orientation)
                    # - Transformation from object frmae to world frame (same as object pose in world frame)
                    wf__T__of = (val["upd_pos"], val["upd_ori"])
                    # - pose of volume frame in world frame:
                    wf__P__vf = p.multiplyTransforms(*wf__T__of, *of__P__vf)

                    # Now we know where the volume (center) should go in world frame.
                    # Let's convert it to kit volume frame (kf stands for kit frame)
                    kf__P__vf = (wf__P__vf[0] - kit_crop_bounds[:, 0], wf__P__vf[1])
                    user_coords = (kf__P__vf[0] / voxel_size).astype(int)
                    user_ori = kf__P__vf[1]

                    # Rotate the p0_vol according to the user provided input
                    rotate_angles = np.array(p.getEulerFromQuaternion(user_ori))
                    # print_ic(rotate_angles)
                    p0_vol_rotate = rotate_tsdf(p0_vol, rotate_angles)
                    p0_vol_rotate_ten = torch.tensor(p0_vol_rotate, device=device).unsqueeze(dim=0)
                    p1_vol_ten = torch.tensor(p1_vol, device=device).unsqueeze(dim=0)
                    # print_ic(user_coords, user_ori)
                    user_coords_ten = torch.tensor(user_coords, device=device).unsqueeze(dim=0)
                    batch = {
                        "p0_vol": p0_vol_rotate_ten,
                        "p1_vol": p1_vol_ten, 
                        "p1_coords": None,
                        "p1_coords_user": user_coords_ten,
                        "p1_ori": None,
                        "concav_ori": torch.tensor([[0, 0, 0, 1]], device=device),
                        "symmetry": torch.tensor([[-1, -1, -1]]),
                    }
                    pred_coords, pred_ori = None, None
                    with torch.no_grad():
                        since = time.time()
                        print("Starting position prediction ...")
                        _, pred_coords, _, _ = transporter.run(
                            batch, training=False, log=False, calc_loss=False)
                        print(f"\tposition prediction finished in {time.time() - since}")
                        since = time.time()
                        print("Starting rotation prediction ...")
                        batch['p1_coords'] = pred_coords.astype(int)
                        _, _, pred_ori, _ = rotator.run(
                            batch, training=False, log=False, calc_loss=False)
                        print(f"\trotation prediction finished in {time.time() - since}")
                        pred_coords = pred_coords[0]
                        # print_ic(pred_coords, pred_ori)

                        final_pos = pred_coords * voxel_size
                        final_ori = multiply_quat(user_ori, pred_ori)

                        vf__T__of = (name_transformations[name], zero_orientation)
                        final_pos, final_ori = p.multiplyTransforms(
                            final_pos, final_ori,
                            *vf__T__of,
                        )
                        final_pos, final_ori = np.array(final_pos), np.array(final_ori)
                        final_pos += kit_crop_bounds[:, 0]
                        all_snap_pose[name] = (final_pos, final_ori)
                    if DEBUG_SNAP:
                        #print("=======>FIXME<======= using larger voxel size for debugging snap")
                        voxel_size_debug = voxel_size * 3
                        # Let's first dump the kit full volume:
                        TSDFHelper.to_mesh(p1_vol, debug_path / f"{name}_p1_vol.obj", voxel_size_debug)
                        # Now let's align the user provided pose to this volumes space
                        vol_origin = (user_coords - np.array([*p1_vol.shape]) / 2) * voxel_size_debug 
                        TSDFHelper.to_mesh(p0_vol_rotate, debug_path / f"{name}_p0_user_provided.obj", voxel_size_debug, vol_origin=vol_origin)
                        # Now let's dump the pred things:
                        p0_vol_rotate_corrected = rotate_tsdf(p0_vol_rotate, np.array(p.getEulerFromQuaternion(pred_ori)))
                        vol_origin = (pred_coords - np.array([*p1_vol.shape]) / 2) * voxel_size_debug 
                        TSDFHelper.to_mesh(p0_vol_rotate_corrected, debug_path / f"{name}_p0_snap_corrected.obj", voxel_size_debug, vol_origin=vol_origin)

                        # Now let's dump initial p0_vol to final prediction together
                        p0_vol_rotate_initial_to_final = rotate_tsdf(p0_vol, np.array(p.getEulerFromQuaternion(final_ori)))
                        vol_origin = (pred_coords - np.array([*p1_vol.shape]) / 2) * voxel_size_debug 
                        TSDFHelper.to_mesh(p0_vol_rotate_initial_to_final, debug_path / f"{name}_p0_snap_initial_to_final.obj", voxel_size_debug, vol_origin=vol_origin)

                # print("Kit place pose after action snapping: ",
                #     final_pos,
                #     np.array(p.getEulerFromQuaternion(final_ori)) * 180 / np.pi
                # )
                # All right, we are ready now


                # 6DoF Insertion primitive:
                # Align the object in 3D with correct orientation a few cm above the insertion pose
                # Then slowly perform the insertion
                # - over-kit pose: 
                OVER_KIT_HEIGHT = 0.1
                wf__client_place_pos = final_pos  # place pos from client in wf (world frame)
                wf__client_place_ori = final_ori 
                # Transform this pose first to gripper pose
                of__gripper_pos = gp - val["curr_pos"]  # gripper tool tip position in obj frame
                of__gripper_ori = p.getQuaternionFromEuler((0, 0, 0))
                wf__gripper_pos, wf__gripper_ori = p.multiplyTransforms(
                    wf__client_place_pos, wf__client_place_ori, of__gripper_pos, of__gripper_ori)

                # Now apply over kit things 
                of__over_kit_pos = np.array([0, 0, OVER_KIT_HEIGHT])  # over-kit position in of (object frame)
                of__over_kit_ori = p.getQuaternionFromEuler((0, 0, 0))
                wf__over_kit_gripper_pos, wf__over_kit_gripper_ori = p.multiplyTransforms(
                    wf__gripper_pos, wf__gripper_ori, of__over_kit_pos, of__over_kit_ori
                )

                if RUN_ROBOT:
                    def get_rpy(quat):
                        return np.array(p.getEulerFromQuaternion(quat))
                    def get_quat(rpy):
                        return  np.array(p.getQuaternionFromEuler(rpy))
                    wf__over_kit_gripper_ori = get_rpy(wf__over_kit_gripper_ori)
                    wf__gripper_ori = get_rpy(wf__gripper_ori)
                    # print_ic(wf__over_kit_gripper_ori * 180 / np.pi, wf__gripper_ori *
                            #  180 / np.pi, np.isclose(wf__over_kit_gripper_ori, wf__gripper_ori))
                    
                    if CHOOSE_BEST_YAW:
                        best_yaw = -np.pi / 2
                        pick_rpy = np.array([0, 0, 0])
                        place_rpy = wf__over_kit_gripper_ori
                        total_yaw_change = place_rpy[2] - pick_rpy[2]
                        pick_rpy = [*pick_rpy[:2], best_yaw - total_yaw_change / 2]
                        place_rpy = [*place_rpy[:2], best_yaw + total_yaw_change / 2]
                    else:
                        pick_rpy = [0, 0, 0]
                        place_rpy = wf__over_kit_gripper_ori
                    
                    # Pick
                    tool_position = gp
                    print("moving to ", tool_position)
                    robot.set_pose_derived([tool_position[0], tool_position[1], bounds_ws[2, 1] / 2, *pick_rpy], 0.1, 0.1)
                    robot.set_pose_derived([tool_position[0], tool_position[1], tool_position[2], *pick_rpy], 0.1, 0.1)
                    robot.close_gripper()
                    robot.set_pose_derived([tool_position[0], tool_position[1], bounds_ws[2, 1] / 2, *pick_rpy], 0.1, 0.1)
                    # Place
                    # wf__over_kit_gripper_pos = np.array(wf__over_kit_gripper_pos)
                    # wf__over_kit_gripper_ori = np.array(p.getEulerFromQuaternion(wf__over_kit_gripper_ori))
                    # print_ic(wf__over_kit_gripper_pos, np.array(wf__over_kit_gripper_ori) * 180 / np.pi, np.array(wf__gripper_ori) * 180 / np.pi)
                    robot.set_pose_derived([*wf__over_kit_gripper_pos, *place_rpy], 0.1, 0.1) 
                    robot.set_pose_derived([*wf__gripper_pos, *place_rpy], 0.1, 0.1)
                    robot.open_gripper()
                    robot.set_pose_derived([*wf__over_kit_gripper_pos, *place_rpy], 0.1, 0.1) 
                    robot.homej()
            # if DEBUG_SNAP:
            #     cols = ["user", "pred"]
            #     from evaluate.html_vis import visualize_helper
            #     visualize_helper(snap_tasks, debug_path, cols, html_file_name="debug_snap.html")

        def update_scene_from_client(transporter, rotator):
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
            for obj_dict in scene_dict["objects"]:
                scene_dict_proc[obj_dict["name"]] = obj_dict
                
            # Need to get an index 
            # - ok. in the new dict. just find the old item. compare
            # - if any difference. add it to diff dict. that's it
            for obj_dict in update_scene_json["objects"]:
                key = obj_dict["name"]
                if key == "kit":
                    continue
                if key not in scene_dict_proc:
                    print(f"Unknown id {key} received")
                    continue
                
                curr_pos, curr_ori = np.array(scene_dict_proc[key]["position"]), np.array(scene_dict_proc[key]["orientation"]) 
                upd_pos, upd_ori = np.array(obj_dict["position"]), np.array(obj_dict["orientation"]) 

                if (np.abs(upd_pos - curr_pos) > update_delta).any() or (np.abs(upd_ori - curr_ori) > update_delta).any():
                    print("Position changed for ", key)
                    # Transform the difference back to real world frame
                    curr_pos__real, curr_ori__real = transform_to_real(curr_pos, curr_ori)
                    upd_pos__real, upd_ori__real = transform_to_real(upd_pos, upd_ori)
                    diff_dict[key] = {
                        "curr_pos": curr_pos__real,
                        "curr_ori": curr_ori__real,
                        "upd_pos": upd_pos__real,
                        "upd_ori": upd_ori__real,
                    }

            execute_scene_diff(diff_dict, transporter, rotator)
            last_mTime = mTime
            return True

        transporter = None
        rotator = None
        if DO_ACTION_SNAPPING:
            transporter = VolMatchTransport.from_cfg(vm_cfg, cfg.env.voxel_size, load_model=True, log=False)
            rotator = VolMatchRotate.from_cfg(vm_cfg, load_model=True, log=False)
        print("Visit the client site and refresh to update. Click \"Upload Scene\" when ready ...")
        while not update_scene_from_client(transporter, rotator): 
            sleep(1)
        if DO_ACTION_SNAPPING:
            del transporter
            del rotator
            torch.cuda.empty_cache()
        datapoint = {
            'masks': name_mask,
            'vols': name_vols,
            'init_pos': name_transformations,
            'user_pose': all_user_pose,
            'snap_pose': all_snap_pose,
            'depth_image': d,
            'gray': gray
        }
        pickle.dump(datapoint, open(debug_path / f"datapoint.pkl", 'wb'))
        # total_system_time = time.time() - system_start_time
        # print_ic(total_system_time)
        # np.savetxt(debug_path / "total_system_time.txt", [total_system_time])
        print("====================================================\n")
        break
    if RUN_ROBOT:
        robot.close()

if __name__ == "__main__":
    main()
