from .baseEnv import BaseEnv
import numpy as np
import pybullet as p
from .ur5 import UR5
from .utils import CuboidMarker, change_body_color, get_tableau_palette, get_surrounding_cameras, SCENETYPE
import shutil
import os
import utils
import json
from PIL import Image
from random import random, randint, sample, uniform
from pathlib import Path
import json
from omegaconf import DictConfig
from environment.gripper import SuctionGripper
from environment.kitGenerator import KitGenerator
from utils.pointcloud import PointCloud
from utils import get_masked_rgb, get_masked_d, get_finetuned_place_pos, get_pix_size
from utils.rotation import multiply_quat, invert_quat
from environment.camera import SimCameraBase, SimCameraPosition 


class TeleRobotEnv(BaseEnv):
    def __init__(self, cfg:DictConfig, srg=None, gui=False, gripper_class=SuctionGripper, dataset_split:str="train", obj_paths:list=list()):
        super().__init__(gui)

        self.workspace_bounds_kit = np.array(cfg.env.workspace_bounds_kit)
        self.view_bounds_kit = np.array(cfg.env.view_bounds_kit)
        self.camera_bounds_kit = np.array(cfg.env.camera_bounds_kit)
        self.camera_look_at_kit = np.array(cfg.env.look_at_kit)
        self.camera_r_bounds_kit = np.array(cfg.env.camera_r_bounds_kit)
        self.camera_min_theta_kit = np.array(cfg.env.camera_min_theta_kit)
        self.camera_max_theta_kit = np.array(cfg.env.camera_max_theta_kit)

        self.workspace_bounds_objects = np.array(cfg.env.workspace_bounds_objects)
        self.view_bounds_objects = np.array(cfg.env.view_bounds_objects)
        self.camera_bounds_objects = np.array(cfg.env.camera_bounds_objects)
        self.camera_look_at_objects = np.array(cfg.env.look_at_objects)
        self.camera_r_bounds_objects = np.array(cfg.env.camera_r_bounds_objects)
        self.camera_min_theta_objects = np.array(cfg.env.camera_min_theta_objects)
        self.camera_max_theta_objects = np.array(cfg.env.camera_max_theta_objects)

        self.image_size = np.array(cfg.env.image_size)
        self._voxel_size = cfg.env.voxel_size
        self.kit_generator = KitGenerator(cfg.env.kit_width, dataset_split=dataset_split)
        self.globalScaling = cfg.env.globalScaling
        self.use_new_kits = cfg.env.use_new_kits
        self.cropped_vol_shape = np.array(cfg.env.cropped_vol_shape)
        self.pix_size = get_pix_size(self.view_bounds_objects, self.image_size[0]) 

        self.assets_path = Path(__file__).parent.parent / "assets"
        self.plane_id = p.loadURDF("plane.urdf")
        self.ur5 = UR5(self, basePosition=np.array(
            cfg.env.robot_position), gripper_class=gripper_class)
        self.json_scene_path = Path("visualizer/client/static/scenes/json_scene")
        self.client_scene_path = Path("visualizer/server/updated_scene.json")
        if self.client_scene_path.exists():
            self.client_scene_path.unlink()
        self.client_refresh_request_path = Path("visualizer/server/refresh_scene.json")
        self.last_mTime = None
        self.last_refresh_request_time = None


        # Setup primary cameras
        self.camera_kit = self.reset_primary_camera(
            self.camera_look_at_kit, *self.camera_r_bounds_kit, self.camera_min_theta_kit, self.camera_max_theta_kit, self.image_size)
        self.camera_objects = self.reset_primary_camera(
            self.camera_look_at_objects, *self.camera_r_bounds_objects, self.camera_min_theta_objects, self.camera_max_theta_objects, self.image_size)

        # Setup ground truth cameras
        self.gt_cameras_kit = get_surrounding_cameras(
            self.camera_bounds_kit, self.camera_look_at_kit, self.image_size)
        self.gt_cameras_objects = get_surrounding_cameras(
            self.camera_bounds_objects, self.camera_look_at_objects, self.image_size)

        self.label_kit = cfg.env.label_kit
        self.label_object = cfg.env.label_object
        self.obj_body_id_labels = dict()
        self.constraint_ids = list()

        # Setup object paths
        self.obj_paths = obj_paths
        
        # setup planes
        self.plane_dirs_root = self.assets_path / "planes"
        self.plane_dirs = [
            plane_dir
            for plane_dir in self.plane_dirs_root.iterdir()
            if not plane_dir.name.startswith(".")
        ]

        self.colors = [c + [1] for c in get_tableau_palette().tolist()]

        self.srg = srg
        self.srg_is_first = True
        self.scene_dict = None
        self.masks = None
        self.rgbs = None
        self.ds = None
        self.cameras_tmp = None
        self.name_mask_index = None
        self.sc_vols = None
        self.update_delta = 1e-5
            
    @property
    def voxel_size(self):
        print(f"TeleRobotEnv.voxel_size is ", self._voxel_size)
        return self._voxel_size

    def visualize_bounds(self):
        # Visualize bounds:
        self.m1 = CuboidMarker(
            self.workspace_bounds_kit[:, 0], self.workspace_bounds_kit[:, 1])
        self.m2 = CuboidMarker(
            self.view_bounds_kit[:, 0], self.view_bounds_kit[:, 1])
        self.m21 = CuboidMarker(
            self.camera_bounds_kit[:, 0], self.camera_bounds_kit[:, 1])
        self.m3 = CuboidMarker(
            self.workspace_bounds_objects[:, 0], self.workspace_bounds_objects[:, 1], rgb_color=np.array([0, 0, 255]))
        self.m4 = CuboidMarker(
            self.view_bounds_objects[:, 0], self.view_bounds_objects[:, 1], rgb_color=np.array([0, 0, 255]))
        self.m41 = CuboidMarker(
            self.camera_bounds_objects[:, 0], self.camera_bounds_objects[:, 1], rgb_color=np.array([0, 0, 255]))

    def clear_scene(self):
        for const_id in self.constraint_ids:
            p.removeConstraint(const_id)
        self.constraint_ids = list()
        for body_id in self.obj_body_id_labels:
            p.removeBody(body_id)
        self.object_body_id_labels = dict()

    def load_kit(self, kit_path, kit_position:np.ndarray=None, kit_orientation:np.ndarray=None, use_z_zero:bool=True):
        if kit_position is None:
            kit_position = (
                self.workspace_bounds_kit[:, 0] + self.workspace_bounds_kit[:, 1]
            ) / 2
        if kit_orientation is None:
            kit_orientation = p.getQuaternionFromEuler([0, 0, -np.pi + 2*np.pi*np.random.random()])
        if use_z_zero:
            kit_position[2] = 0
            kit_body_id = p.loadURDF(
                str(kit_path),
                basePosition=kit_position, baseOrientation=kit_orientation,
                useFixedBase=True, globalScaling=self.globalScaling
            )
        else:  # Drop kit from a height and constraint it later
            kit_body_id = p.loadURDF(
                str(kit_path),
                basePosition=kit_position, baseOrientation=kit_orientation,
                globalScaling=self.globalScaling
            )
            self.step_simulation(100)  # let the kit fall and settle
            fix_kit_pos, fix_kit_ori =p.getBasePositionAndOrientation(kit_body_id)
            fix_kit_pos = np.array(fix_kit_pos)
            fix_kit_pos[2] += 0.005
            const_id = p.createConstraint(
                parentBodyUniqueId=kit_body_id,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=fix_kit_pos,
                childFrameOrientation=fix_kit_ori,
            )
            self.constraint_ids.append(const_id)

        change_body_color(kit_body_id, sample(self.colors, 1)[0])
        self.obj_body_id_labels[kit_body_id] = self.label_kit
        self.step_simulation(500)
        return kit_body_id

    def load_obj(self, obj_path, bounds):
        position = bounds[:, 0] + np.random.random((3,)) * (bounds[:, 1] - bounds[:, 0])
        position[2] = bounds[2, 1]
        orientation = p.getQuaternionFromEuler(-np.pi + 2*np.pi*np.random.random((3,)))
        obj_body_id = p.loadURDF(str(obj_path), basePosition=position, baseOrientation=orientation, globalScaling=self.globalScaling)
        change_body_color(obj_body_id, rgbaColor= sample(self.colors, 1)[0])
        self.step_simulation(5e2)
        self.obj_body_id_labels[obj_body_id] = self.label_object
        return obj_body_id

    def reset_plane(self):
        p.removeBody(self.plane_id)
        plane_dir = sample(self.plane_dirs, 1)[0]
        self.plane_id = p.loadURDF(str(plane_dir / "plane.urdf"))

    def reset_scene(self, output_dir:Path, load_single_object=False):
        self.clear_scene()
        self.reset_plane()
        kit_urdf = output_dir / "kit.urdf"
        obj_details = self.kit_generator.generate_random_kit(kit_urdf)
        kit_body_id = self.load_kit(str(kit_urdf))
        for obj_detail in obj_details:
            obj_detail["body_id"] = self.load_obj(obj_detail["path"], self.workspace_bounds_objects)
            if load_single_object:
                break
        return self.obj_body_id_labels, obj_details, kit_body_id

    def reset_scene_random(self, output_dir, scene_prob_dist=np.ones((3,)) * 1 / 3, six_dof:bool=False):
        self.clear_scene()
        self.reset_plane()
        scene_type = SCENETYPE(np.random.choice(
            np.arange(len(SCENETYPE), dtype=np.int), p=scene_prob_dist))

        if scene_type == SCENETYPE.KIT or scene_type == SCENETYPE.KIT_OBJECTS:
            # load a kit
            kit_path = output_dir / "kit.urdf"
            if six_dof:
                self.kit_generator.generate_random_kit_3d_five_plates(kit_path)
                self.load_kit(kit_path, use_z_zero=False)
            else:
                self.kit_generator.generate_random_kit(kit_path)
                self.load_kit(kit_path)

        if scene_type != SCENETYPE.KIT:
            # A kit has maximum 5 holes and hence max 5 objects
            num_scene_objects = sample(range(5 + 1), 1)[0]
            # make sure atleast one object is present in scene
            if num_scene_objects == 0 and len(self.obj_body_id_labels) == 0:
                num_scene_objects = 1

            for _ in range(num_scene_objects):
                obj_path = sample(self.obj_paths, 1)[0]
                bounds = self.workspace_bounds_objects if scene_type == SCENETYPE.OBJECTS else self.workspace_bounds_kit
                self.load_obj(obj_path, bounds)
        return self.obj_body_id_labels, scene_type

    def reset_scene_6DoF(self, output_dir, unit_kit:bool=True, load_single_object:bool=True):
        self.clear_scene()
        self.reset_plane()
        # Kit
        kit_path = output_dir / "kit.urdf"
        if unit_kit:
            obj_details = self.kit_generator.generate_random_kit_3d_one_plate(kit_path)
            kit_position =  self.workspace_bounds_kit[:, 0] + np.random.random((3,)) * (self.workspace_bounds_kit[:, 1] - self.workspace_bounds_kit[:, 0])
            kit_orientation = np.array([0, 0, 0, 1])
            kit_body_id = self.load_kit(kit_path, kit_position, kit_orientation)
        else:
            obj_details = self.kit_generator.generate_random_kit_3d_five_plates(kit_path)
            kit_body_id = self.load_kit(kit_path, use_z_zero=False)  
        # Object
        for i in range(len(obj_details)): 
            obj_details[i]["body_id"] = self.load_obj(obj_details[i]["path"], self.workspace_bounds_objects)
            if load_single_object:
                break
        return self.obj_body_id_labels, obj_details, kit_body_id

    def save_environment_ground_truth(self):
        # save ground truth environment information 
        # I have tote and shapes in the environment
        scene_dict = dict()
        scene_dict["objects"] = list()

        # save kit 
        shutil.copy("assets/kitting/kit.obj", self.json_scene_path)
        pose = p.getBasePositionAndOrientation(self.kit_body_id)
        obj_dict = {
            "name": "kit",
            "path": "kit.obj",
            "position": [pose[0][0], pose[0][1], 0],
            "orientation": list(pose[1]),
        }
        scene_dict["objects"].append(obj_dict)

        for i, id in enumerate(self.object_body_ids):
            obj_path = utils.change_extension(self.object_paths[i], ".obj")
            shutil.copy(obj_path, f"{self.json_scene_path}/{id}.obj")
            pose = p.getBasePositionAndOrientation(id)
            obj_dict = {
                "name": id,
                "path": f"{id}.obj",
                "position": list(pose[0]),
                "orientation": list(pose[1])
            }
            scene_dict["objects"].append(obj_dict)
        with open(f"{self.json_scene_path}/scene.json", "w") as f:
            json.dump(scene_dict, f)
        
        # scene image
        rgb_img, depth_img = self.camera.get_image()
        Image.fromarray(rgb_img).save(self.json_scene_path + "/scene_rgb.png")

        # scene pt cloud
        cam_pts = depth_to_point_cloud(self.camera.intrinsics, depth_img)
        pts = transform_point3s(self.camera.pose_matrix, cam_pts)
        if len(pts) == 0:
            print("Empty point cloud!")
        else:
            ptcloud = PointCloud(vertices=pts, colors=[0, 255, 0])  # Green
            ptcloud.export(self.json_scene_path + "/scene_pt_cloud.ply")
    
    def upload_scene_to_client(self):
        if self.obj_body_id_labels is None:
            print("Please reset scene")
            return
        self.scene_dict, self.masks, self.rgbs, self.ds, self.cameras_tmp, self.name_mask_index, self.sc_vols = self.srg.dump_sr(
            self, self.obj_body_id_labels, self.srg_is_first)
        with open(self.json_scene_path/"scene.json", "w") as f:
            json.dump(self.scene_dict, f)

        # scene pt cloud
        def dump_scene(scene_type):
            pcl = self.get_scene_pcl(scene_type)
            pcl.export(self.json_scene_path / f"scene_{scene_type.name}_pcl.ply")
        dump_scene(SCENETYPE.OBJECTS)
        dump_scene(SCENETYPE.KIT)

        self.srg_is_first = False
    
    def update_scene_from_client(self):
        if not self.client_scene_path.exists():
            return False
        mTime = self.client_scene_path.stat().st_mtime
        if self.last_mTime is not None and self.last_mTime == mTime:
            return False

        with open(self.client_scene_path, "r") as f:
            update_scene_json = json.load(f)
        print("Updating scene")
        diff_dict = dict()

        scene_dict_proc = dict()
        for obj_dict in self.scene_dict["objects"]:
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

            if (np.abs(upd_pos - curr_pos) > self.update_delta).any() or (np.abs(upd_ori - curr_ori) > self.update_delta).any():
                print("Position changed for ", key)
                diff_dict[key] = {
                    "curr_pos": curr_pos,
                    "curr_ori": curr_ori,
                    "upd_pos": upd_pos,
                    "upd_ori": upd_ori,
                }

        self.execute_scene_diff(diff_dict)
        self.last_mTime = mTime
        return True
    
    def execute_scene_diff(self, diff_dict):
        # For every object in diff dict. Just move them one by one
        pose1_vol = self.sc_vols[1]["kit"] if "kit" in self.sc_vols[1] else self.sc_vols[0]["kit"]
        for name, val in diff_dict.items():
            # Cool. Now just use mask to get valid grasping position
            (i, ind) = self.name_mask_index[name]
            rgb = self.rgbs[i]
            d = self.ds[i]
            camera = self.cameras_tmp[i]
            mask = self.masks[i][ind]
            masked_rgb = get_masked_rgb(mask, rgb)
            masked_d = get_masked_d(mask, d)
            # Estimate Normals from this depth image
            pc = PointCloud(masked_rgb, masked_d, camera.intrinsics)  # note RGB image can be grayscale
            pc.make_pointcloud(extrinsics=camera.pose_matrix, trim=True)
            pc.compute_normals()
            ns = pc.normals
            point_cloud = pc.point_cloud
            valid_indices = ns[:, 2] > 0.99
            if len(valid_indices) == 0:
                valid_indices = ns[:, 2] > ns[:, 2].max() - 0.1

            # Find point nearest to the center  
            pc1 = point_cloud[valid_indices][:, :3]
            x_mid = (pc1[:, 0].max() + pc1[:, 0].min()) / 2
            y_mid = (pc1[:, 1].max() + pc1[:, 1].min()) / 2
            gp = np.array([x_mid, y_mid, pc1[:, 2].mean()])
            # Correct approach will be to use the corresponding mask and find a flat surface
            gs = self.ur5.execute_grasp(gp, 0, self.obj_body_id_labels.keys())
            if gs:
                pose0_vol = self.sc_vols[i][name]
                pose1_pos = np.array(val["upd_pos"])

                pose0_width = self.cropped_vol_shape * self.voxel_size
                position, angle = get_finetuned_place_pos(
                    pose0_vol, pose1_vol, pose1_pos, pose0_width, self.view_bounds_kit, self.voxel_size)
                self.ur5.execute_place(position, angle)

    def refresh_client_scene(self):
        # check if client has asked for it
        mTime = None
        if os.path.exists(self.client_refresh_request_path):
           mTime = os.path.getmtime(self.client_refresh_request_path)
        else:
            return
        if mTime is not None and self.last_refresh_request_time == mTime:
            return
        self.last_refresh_request_time = mTime
        self.save_environment_ground_truth()
        print("Client scene refreshed")
    
    def get_view_bounds(self, scene_type):
        view_bounds = self.view_bounds_kit\
            if scene_type == SCENETYPE.KIT or scene_type == SCENETYPE.KIT_OBJECTS\
            else self.view_bounds_objects
        return view_bounds

    def get_gt_cameras(self, scene_type):
        gt_cameras = self.gt_cameras_kit\
            if scene_type == SCENETYPE.KIT or scene_type == SCENETYPE.KIT_OBJECTS\
            else self.gt_cameras_objects
        return gt_cameras

    def get_camera(self, scene_type: SCENETYPE) -> SimCameraBase:
        camera = self.camera_kit\
            if scene_type == SCENETYPE.KIT or scene_type == SCENETYPE.KIT_OBJECTS\
            else self.camera_objects
        return camera

    def get_scene_volume(self, scene_type, use_surrounding_cameras=False, return_first_image=False):
        # Cool. Just figure out the cameras and that's it
        view_bounds = self.get_view_bounds(scene_type)
        cameras  = [self.get_camera(scene_type)]
        if use_surrounding_cameras:
            cameras += self.get_gt_cameras(scene_type)
        return super().get_scene_volume(cameras=cameras, view_bounds=view_bounds, return_first_image=return_first_image)

    def get_scene_pcl(self, scene_type: SCENETYPE):
        return super().get_scene_pcl(self.get_camera(scene_type))

    def get_scene_cmap_hmap(self, scene_type: SCENETYPE):
        cameras = self.get_cameras(scene_type)
        bounds = self.get_view_bounds(scene_type) 
        return super().get_scene_cmap_hmap(cameras, bounds, self.pix_size)
    
    def get_gt_pick_place(self, obj_detail, kit_body_id):
        obj_pose_world = p.getBasePositionAndOrientation(obj_detail["body_id"])
        targ_pose_kit = obj_detail["position"], p.getQuaternionFromEuler(obj_detail["orientation"])
        kit_pose_world = p.getBasePositionAndOrientation(kit_body_id)
        # targ_pose_world = p.multiplyTransforms(*targ_pose_kit, *kit_pose_world)
        # obj_to_targ_ori = multiply_quat(targ_pose_world[1], invert_quat(obj_pose_world[1]))

        obj_to_world_ori = p.invertTransform((0, 0, 0), obj_pose_world[1])
        obj_to_kit = p.multiplyTransforms(*targ_pose_kit, *obj_to_world_ori)
        targ = p.multiplyTransforms( [0, 0, 0], kit_pose_world[1], *obj_to_kit )

        concav_ori = p.multiplyTransforms(*kit_pose_world, *targ_pose_kit)[1]

        return obj_pose_world[0], np.array(kit_pose_world[0])+np.array(targ[0]), targ[1], concav_ori

    @staticmethod
    def reset_primary_camera(look_at, min_r, max_r, min_theta, max_theta, image_size):
        r = uniform(min_r, max_r)
        theta = uniform(min_theta, max_theta)
        phi = uniform(0, 2 * np.pi)
        eyePosition = np.array([
            r*np.sin(theta)*np.cos(phi),
            r*np.sin(theta)*np.sin(phi),
            r*np.cos(theta)
        ]) + look_at
        uv = [0, 0, 1]
        return SimCameraPosition(eyePosition, look_at, uv, image_size=image_size)  # Main camera
    
    @staticmethod
    def get_gt_camera(r=0.9, look_at = np.array([0.35, 0, 0.2])):
        theta = 0.9
        phi = 0
        eyePosition = np.array([
            r*np.sin(theta)*np.cos(phi),
            r*np.sin(theta)*np.sin(phi),
            r*np.cos(theta)
        ]) + look_at
        uv = [0, 0, 1]
        image_size = np.array([720, 1280])
        return SimCameraPosition(eyePosition, look_at, uv, image_size=image_size)  # Main camera