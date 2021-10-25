# This file is for testing workspace bounds
import sys
from pathlib import Path
from time import sleep
root_path = Path(__file__).parent.parent.absolute()
sys.path.append(str(root_path))

from environment.real.ur5 import UR5_URX
import numpy as np
from real_world.utils import get_kit_bounds, get_obj_bounds, get_obj_masks, get_tool_init, get_workspace_bounds
from real_world.calibrate import get_calibration_bounds


# rows: x,y,z; cols: min,max
def test_bounds(robot: UR5_URX, bounds):
    for z in bounds[2, :]:
        for y in bounds[1, :]:
            for x in bounds[0, :]:
                tool_position = np.array([x, y, z])
                print("moving to ", tool_position)
                robot.set_pos_derived(tool_position + [0, 0, 0.05], )
                robot.set_pos_derived(tool_position)
                input("waiting for user...")
                robot.set_pos_derived(tool_position + [0, 0, 0.05])
        break

def main():
    tool_offset, _ = get_tool_init()
    workspace_bounds = get_workspace_bounds()

    robot = UR5_URX(j_vel=0.2, j_acc=0.2, tool_offset=tool_offset)
    robot.homej()
    # print("calibration")
    # cal_bounds = get_calibration_bounds()
    # test_bounds(robot, cal_bounds)
    print("object")
    object_bounds = get_obj_bounds()
    test_bounds(robot, object_bounds)
    print("kit")
    kit_bounds = get_kit_bounds()
    test_bounds(robot, kit_bounds)
    robot.homej()

if __name__ == "__main__":
    main()
