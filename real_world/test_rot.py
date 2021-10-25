import sys
from pathlib import Path
root_path = Path(__file__).parent.parent.absolute()
# print("Adding to python path: ", root_path)
sys.path = [str(root_path)] + sys.path

import hydra
from omegaconf import DictConfig
from environment.real.ur5 import UR5_URX
from real_world import utils as rw_utils
import numpy as np
import math3d as m3d


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    tool_offset, tool_orientation = rw_utils.get_tool_init()
    tool_offset = [0.0, -0.1, 0.270, 0, np.pi, 0]
    j_acc = 0.2
    robot = UR5_URX(j_vel=j_acc, j_acc=j_acc, tool_offset=tool_offset)
    robot.homej()
    obj_bounds = rw_utils.get_obj_bounds()
    pt = (obj_bounds[:, 0] + obj_bounds[:, 1]) / 2
    print("moving to ", pt)

    robot.set_pos_derived(pt)

    # Visualize x direction
    def visualize_directions(direc):
        input("Waiting for user input ...")
        robot.set_pos_derived(direc * 0.1 + pt) 
        robot.set_pos_derived(pt) 
    # print("X Directiion:") 
    # visualize_directions(np.array([1, 0, 0]))
    # print("Y Directiion:") 
    # visualize_directions(np.array([0, 1, 0]))
    # print("Z Directiion:") 
    # visualize_directions(np.array([0, 0, 1]))

    robot.add_pose_tool(m3d.Transform([0, 0, 0, 0, np.pi / 4, 0]), j_acc, j_acc)
    robot.add_pose_tool(m3d.Transform([0, 0, 0, 0, 0, np.pi / 4]), j_acc, j_acc)
    robot.add_pose_tool(m3d.Transform([0, 0, -0.2, 0, 0, 0]), j_acc, j_acc)

    n = 2
    for i in range(3):
        print("\n\n===========")
        print(f"Visualizing {i} orientation")
        ori = np.zeros(3)
        # robot.set_pose_derived([*pt, *ori])
        # robot.set_pos_derived(pt)
        for angle in np.linspace(-np.pi/8, np.pi/8, n):
            # ori[i] = angle
            input(f"Press enter to move to ori {ori}...")
            # robot.set_pose_derived([*pt, *ori])
            ori[i] = np.pi / 20
            robot.add_pose_tool(m3d.Transform([0, 0, 0, *ori]))
        robot.add_pose_tool(m3d.Transform([0, 0, 0, *(-n*ori)]))

    robot.homej()




if __name__ == "__main__":
    main()