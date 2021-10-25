import sys
from pathlib import Path
root_path = Path(__file__).parent.parent.absolute()
# print("Adding to python path: ", root_path)
sys.path = [str(root_path)] + sys.path

import time
import numpy as np
import cv2
from environment.real.ur5 import UR5_URX
from environment.real.cameras import RealSense
from real_world.utils import get_tool_init, get_workspace_bounds


if __name__ == "__main__":
    # rows: x,y,z; cols: min,max
    tool_offset, _ = get_tool_init()
    workspace_bounds = get_workspace_bounds()

    robot = UR5_URX(j_vel=0.2, j_acc=0.2, tool_offset=tool_offset)
    robot.homej()
    input("Running touch script. Press Enter to continue...")

    # See https://github.com/columbia-robovision/PyRealSense for more details
    bin_cam = RealSense()

    # Callback function for clicking on OpenCV window
    click_point_pix = ()
    color_im, depth_im = bin_cam.get_camera_data(avg_depth=False)

    click_num = 0
    def mouseclick_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            global click_num
            global click_point_pix
            click_point_pix = (x, y)

            # Get click point in camera coordinates
            click_z = depth_im[y, x]
            click_x = (x-bin_cam.color_intr[0, 2]) * \
                click_z/bin_cam.color_intr[0, 0]
            click_y = (y-bin_cam.color_intr[1, 2]) * \
                click_z/bin_cam.color_intr[1, 1]
            if click_z == 0:
                return
            click_point = np.asarray([click_x, click_y, click_z])
            click_point = np.append(click_point, 1.0).reshape(4, 1)

            # Convert camera coordinates to robot coordinates
            target_position = np.dot(bin_cam.pose, click_point)
            target_position = target_position[0:3, 0]
            print(target_position)

            # Move robot to target position
            robot.set_pos_derived([target_position[0], target_position[1], workspace_bounds[2, :].mean()])
            robot.set_pos_derived(target_position)
            # time.sleep(3.0)
            print("Press any key to move the robot back...")
            input()
            # Move robot back to home position
            robot.homej()
            click_num += 1

    # Show color and depth frames
    cv2.namedWindow('color')
    cv2.setMouseCallback('color', mouseclick_callback)

    while True:
        color_im, depth_im = bin_cam.get_camera_data(avg_depth=True, avg_over_n=10)
        bgr_data = cv2.cvtColor(color_im, cv2.COLOR_RGB2BGR)
        if len(click_point_pix) != 0:
            bgr_data = cv2.circle(bgr_data, click_point_pix, 7, (0, 0, 255), 2)
        cv2.imshow('color', bgr_data)

        if cv2.waitKey(1) == ord('c'):
            break
