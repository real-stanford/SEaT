# This file contains code for kit and object bounds in image space
import sys
from pathlib import Path
root_path = Path(__file__).parent.parent.absolute()
sys.path.append(str(root_path))

import numpy as np
from environment.real.cameras import RealSense
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.ndimage import rotate
from real_world.rw_utils import get_workspace_bounds, get_crops, transform_world_to_camera_multi


def plot_rectangle(points_uv, color, ax):
    for i in range(len(points_uv)):
        u1, v1 = points_uv[i]
        u2, v2 = points_uv[(i + 1) % 4]
        ax.plot([u1, u2], [v1, v2], color=color)


def plot_rectangle_vertices(points, ax):
    colors = ["blue", "yellow", "white", "pink"]
    for i, uv in enumerate(points):
        circ = patches.Circle((uv[0], uv[1]), radius=5, color=colors[i])
        ax.add_patch(circ)
        print(f"{colors[i]}: {uv}")


def main():
    cam_pose = np.loadtxt("real_world/camera_pose.txt")
    bin_cam = RealSense(tcp_ip='127.0.0.1', tcp_port=50010,
                        im_h=720, im_w=1280, max_depth=3.0)

    workspace_bounds = get_workspace_bounds()
    # Get vertices for object and kit bounds
    x_min, x_max = workspace_bounds[0, :]
    y_min, y_max = workspace_bounds[1, :]
    z_min, _ = workspace_bounds[2, :]
    x_mid, _, _  = (workspace_bounds[:, 0] + workspace_bounds[:, 1]) / 2
    points_objects = np.array([
        [x_max, y_min, z_min, 1],
        [x_mid, y_min, z_min, 1],
        [x_mid, y_max, z_min, 1],
        [x_max, y_max, z_min, 1],
    ])
    points_kit = np.array([
        [x_mid, y_min, z_min, 1],
        [x_min, y_min, z_min, 1],
        [x_min, y_max, z_min, 1],
        [x_mid, y_max, z_min, 1],
    ])
    points_objects_uv = transform_world_to_camera_multi(points_objects, cam_pose, bin_cam.color_intr)
    points_kit_uv = transform_world_to_camera_multi(points_kit, cam_pose, bin_cam.color_intr)

    # Visualize kit and obj bounds in oroginal camera image
    color_im = bin_cam.color_im
    _, ax = plt.subplots(1,1)
    ax.imshow(color_im)
    plot_rectangle(points_objects_uv, "green", ax)
    plot_rectangle(points_kit_uv, "red", ax)
    plot_rectangle_vertices(points_objects_uv, ax)

    obj_crop, _, _ = get_crops(color_im, points_objects_uv)
    kit_crop, _, _ = get_crops(color_im, points_kit_uv)
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(obj_crop)
    ax[1].imshow(kit_crop)
    plt.show()


if __name__ == "__main__":
    main()