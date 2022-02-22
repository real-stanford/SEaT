import sys
from pathlib import Path
root_path = Path(__file__).parent.parent.absolute()
sys.path = [str(root_path)] + sys.path
import numpy as np
from environment.real.cameras import RealSense
import time
from time import sleep
import hydra
from omegaconf import DictConfig
import json
from environment.real.ur5 import UR5_URX
import pybullet as p
from real_world.rw_utils import get_kit_bounds, get_obj_bounds, get_workspace_bounds, get_tool_init,\
    get_client_frame_pose
from environment.utils import SCENETYPE
from utils.pointcloud import PointCloud
import open3d as o3d
from utils import mkdir_fresh
from icecream import ic as print_ic
from real_world.pyphoxi import PhoXiSensor
import cv2

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # bin_cam = RealSense()
    # camera_pose = bin_cam.pose
    # camera_color_intr = bin_cam.depth_intr
    tcp_ip = "127.0.0.1"
    tcp_port = 50200
    bin_cam = PhoXiSensor(tcp_ip, tcp_port)
    bin_cam.start()
    camera_pose = bin_cam._extr
    camera_color_intr = bin_cam._intr
    bounds_ws = get_workspace_bounds()
    bounds_obj = get_obj_bounds()
    bounds_kit = get_kit_bounds()

    scene_path = mkdir_fresh(Path(cfg.perception.scene_path))
    last_mTime = 0
    client_scene_path = Path("visualizer/server/updated_scene.json")
    if client_scene_path.exists():
        client_scene_path.unlink()

    # Get Real (robot frame) <-> Client transformations
    # - Client pose in real frame
    client_pos__real, client_ori__real = get_client_frame_pose()
    client_ori__real = p.getQuaternionFromEuler(client_ori__real)
    # - Transformation from client to real
    rot_mat = np.array(p.getMatrixFromQuaternion(
        client_ori__real)).reshape((3, 3))
    real__T__client = np.eye(4)
    real__T__client[:3, :3] = rot_mat
    real__T__client[:3, 3] = client_pos__real
    # - Transformation from real to client
    client__T__real = np.linalg.inv(real__T__client)

    def transform_to_real(pos__client, ori__client):
        pos__real, ori__real = p.multiplyTransforms(
            client_pos__real, client_ori__real, pos__client, ori__client)
        return list(pos__real), list(ori__real)

    # Setup robot
    tool_offset, _ = get_tool_init()
    robot = UR5_URX(j_vel=0.3, j_acc=0.3, tool_offset=tool_offset)
    print("Moving robot to home")
    robot.homej()
    debug_path_name = cfg.perception.debug_path_name
    debug_root = Path(f"real_world/debug/{debug_path_name}/baseline")
    debug_root.mkdir(exist_ok=True, parents=True)
    debug_ind = len(list(debug_root.glob('*')))
    debug_path = debug_root / f'T{debug_ind}'
    debug_path.mkdir()

    def prevent_z_collision(trans: np.ndarray) -> np.ndarray:
        z = trans[2]
        # object height is 5 cm. grippper's z should be greater than that
        z_min = bounds_ws[2, 0] + 0.06
        if z < z_min:
            print(f"Clipping z from {z} to {z_min} to prevent collision")
            z = z_min
        trans[2] = z
        return trans

    while True:
        input("Please set scene and press ENTER ....")
        print("Please enter the save folder name for user:")
        debug_path = mkdir_fresh(debug_root / debug_path_name, ask_first=False)
        print_ic(debug_path)

        system_start_time = time.time()
        # rgb, d = bin_cam.get_camera_data(avg_depth=True, avg_over_n=50)
        _, gray, d = bin_cam.get_frame(True)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        pcl = PointCloud(rgb, d, camera_color_intr)
        pcl.make_pointcloud()
        o3d_pc_full = pcl.o3d_pc
        o3d_pc_full = o3d_pc_full.transform(camera_pose)

        def save_point_cloud(bounds, pcl_path):
            # Delete old point cloud first
            if pcl_path.exists():
                pcl_path.unlink()
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

        save_point_cloud(bounds_obj, scene_path /
                         f"scene_{SCENETYPE.OBJECTS.name}_pcl.ply")
        save_point_cloud(bounds_kit, scene_path /
                         f"scene_{SCENETYPE.KIT.name}_pcl.ply")

        def diff_quat(q1, q2):
            e1 = np.array(p.getEulerFromQuaternion(q1))
            e2 = np.array(p.getEulerFromQuaternion(q2))
            return np.array(p.getQuaternionFromEuler(e1 - e2))

        def execute_scene_diff(diff_dict):
            val = {
                "curr_pos": diff_dict["pick"]["pos"],
                "curr_ori": diff_dict["pick"]["ori"],
                "upd_pos": diff_dict["place"]["pos"],
                "upd_ori": diff_dict["place"]["ori"],
            }
            # Same primitive here as well
            def execute_primitive(pose, do_open, over_height=0.15, do_clip: bool=False):
                of__gripper_pos = np.array([0, 0, over_height])
                of__gripper_ori = np.array([0, 0, 0, 1])
                wf__gripper_pose = p.multiplyTransforms(*pose, of__gripper_pos, of__gripper_ori)
                trans = [*wf__gripper_pose[0], *p.getEulerFromQuaternion(wf__gripper_pose[1])]
                robot.set_pose_derived(trans)
                pose = [*pose[0], *p.getEulerFromQuaternion(pose[1])]
                pose = prevent_z_collision(pose) if do_clip else pose
                robot.set_pose_derived(pose)
                if do_open:
                    robot.open_gripper()
                else:
                    robot.close_gripper()
                robot.set_pose_derived(trans, 0.1, 0.1)

            def get_rpy(quat):
                return np.array(p.getEulerFromQuaternion(quat))

            def get_quat(rpy):
                return  np.array(p.getQuaternionFromEuler(rpy))

            # Here I am testing the prevention of robot self collision technique.
            best_yaw = -np.pi / 2
            pick_rpy = get_rpy(val["curr_ori"])
            place_rpy = get_rpy(val["upd_ori"])
            total_yaw_change = place_rpy[2] - pick_rpy[2]
            pick_quat = get_quat([*pick_rpy[:2], best_yaw - total_yaw_change / 2])
            place_quat = get_quat([*place_rpy[:2], best_yaw + total_yaw_change / 2])

            execute_primitive((val["curr_pos"], pick_quat), do_open=False)
            execute_primitive((val["upd_pos"], place_quat), do_open=True, do_clip=True)
            robot.homej()


        def update_scene_from_client():
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

            # Need to get an index
            # - ok. in the new dict. just find the old item. compare
            # - if any difference. add it to diff dict. that's it
            for val in update_scene_json["objects"]:
                # Transform the difference back to real world frame
                pos = np.array(val["position"])
                ori = np.array(val["orientation"])
                pos__real, ori__real = transform_to_real(pos, ori)
                diff_dict[val["name"]] = {
                    "pos": np.array(pos__real),
                    "ori": np.array(ori__real),
                }

            execute_scene_diff(diff_dict)
            last_mTime = mTime
            return True

        print("Waiting for client to update scene ...")
        while not update_scene_from_client():
            sleep(1)
        print("\tMade changes from client!")
        total_system_time = time.time() - system_start_time
        print_ic(total_system_time)
        np.savetxt(debug_path / "total_system_time.txt", [total_system_time])
        print("\n\n====================================================\n")
        break
    robot.close()


if __name__ == "__main__":
    main()