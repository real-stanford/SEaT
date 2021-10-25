import sys
from pathlib import Path
from time import sleep
root_path = Path(__file__).parent.parent.absolute()
sys.path.append(str(root_path))
import time
import numpy as np
import cv2
from scipy import optimize
from environment.real.ur5 import UR5_URX
from environment.real.cameras import RealSense
from real_world.utils import get_tool_init, get_workspace_bounds


def get_calibration_bounds():
    return np.array([
        [-0.58,  -0.48],
        [-0.23, 0.05],
        [0.02, 0.258]
    ])

if __name__ == "__main__":
    # rows: x,y,z; cols: min,max
    tool_offset, tool_orientation = get_tool_init()
    workspace_bounds = get_calibration_bounds()

    # Connect to the camera (see https://github.com/columbia-robovision/PyRealSense for more details)
    bin_cam = RealSense(cam_pose_path=None, cam_depth_scale_path=None)

    # Tool offset from tip of UR5's last joint
    robot = UR5_URX(j_vel=0.5, j_acc=0.5, tool_offset=tool_offset)
    robot.homej()
    input("Running calibration. Press Enter to continue...")

    # Constants
    calib_grid_step = 0.03
    checkerboard_offset = np.array([-(0.03356 + 0.001) / 2, 0, tool_offset[2] - 0.09])

    # Construct 3D calibration grid across workspace
    gridspace_x = np.linspace(workspace_bounds[0, 0], workspace_bounds[0, 1], 1 + int(
        (workspace_bounds[0, 1]-workspace_bounds[0, 0])/calib_grid_step))
    gridspace_y = np.linspace(workspace_bounds[1, 0], workspace_bounds[1, 1], 1+int(
        (workspace_bounds[1, 1]-workspace_bounds[1, 0])/calib_grid_step))
    calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(
        gridspace_x, gridspace_y, workspace_bounds[2, 0]+0.1)
    num_calib_grid_pts = calib_grid_x.shape[0] * \
        calib_grid_x.shape[1]*calib_grid_x.shape[2]
    calib_grid_x.shape = (num_calib_grid_pts, 1)
    calib_grid_y.shape = (num_calib_grid_pts, 1)
    calib_grid_z.shape = (num_calib_grid_pts, 1)
    calib_grid_pts = np.concatenate(
        (calib_grid_x, calib_grid_y, calib_grid_z), axis=1)

    color_im, _ = bin_cam.get_camera_data(avg_depth=False, avg_over_n=10)
    vis_im = cv2.circle(
        color_im, (1, 1), 7, (0, 255, 0), 2)
    cv2.imshow('Calibration', cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR))
    cv2.waitKey(10)

    # Move robot to each calibration point in workspace
    measured_pts = list()
    observed_pts = list()
    observed_pix = list()

    for calib_pt_idx in range(num_calib_grid_pts):
        tool_position = calib_grid_pts[calib_pt_idx, :]
        tool_position[2] = workspace_bounds[2, 0]
        print("Moving robot to: ", tool_position, tool_orientation)
        robot.set_pos_derived(tool_position, acc=0.1, vel=0.1)
        time.sleep(2.0)

        while True:
            color_im, depth_im = bin_cam.get_camera_data(avg_depth=True, avg_over_n=10)
            chckr_size = (3, 3)
            refine_criteria = (cv2.TERM_CRITERIA_EPS +
                               cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            bgr_im = cv2.cvtColor(color_im, cv2.COLOR_RGB2BGR)
            gray_im = cv2.cvtColor(bgr_im, cv2.COLOR_RGB2GRAY)
            chckr_found, crnrs = cv2.findChessboardCorners(
                gray_im, chckr_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH)
            if chckr_found:
                crnrs_refined = cv2.cornerSubPix(
                    gray_im, crnrs, (3, 3), (-1, -1), refine_criteria)
                block_pix = crnrs_refined[4, 0, :]
                break
            time.sleep(0.01)

        # Get observed checkerboard center 3D point in camera space
        block_z = depth_im[
            int(np.round(block_pix[1])),
            int(np.round(block_pix[0]))
        ]
        block_x = np.multiply(
            block_pix[1] - bin_cam.color_intr[0, 2],
            block_z / bin_cam.color_intr[0, 0]
        )
        block_y = np.multiply(
            block_pix[0] - bin_cam.color_intr[1, 2],
            block_z / bin_cam.color_intr[1, 1]
        )
        if block_z == 0:
            print("block_z 0. Continuing...")
            continue

        # Save calibration point and observed checkerboard center
        observed_pts.append([block_x, block_y, block_z])
        tool_position += checkerboard_offset
        measured_pts.append(tool_position)
        observed_pix.append(block_pix)

        # Draw and display the corners
        vis_im = cv2.circle(
            color_im, (int(block_pix[0]), int(block_pix[1])), 7, (0, 255, 0), 2)
        cv2.imshow('Calibration', cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR))
        cv2.waitKey(10)

    # Move robot back to home pose
    robot.homej()

    measured_pts = np.asarray(measured_pts)
    observed_pts = np.asarray(observed_pts)
    observed_pix = np.asarray(observed_pix)
    world2camera = np.eye(4)

    # Estimate rigid transform with SVD (from Nghia Ho)
    def get_rigid_transform(A, B):
        assert len(A) == len(B)
        N = A.shape[0]  # Total points
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - np.tile(centroid_A, (N, 1))  # Centre the points
        BB = B - np.tile(centroid_B, (N, 1))
        # Dot is matrix multiplication for array
        H = np.dot(np.transpose(AA), BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:  # Special reflection case
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        t = np.dot(-R, centroid_A.T) + centroid_B.T
        return R, t

    def get_rigid_transform_error(z_scale):
        global measured_pts, observed_pts, observed_pix, world2camera

        # Apply z offset and compute new observed points using camera intrinsics
        observed_z = observed_pts[:, 2:] * z_scale
        observed_x = np.multiply(
            observed_pix[:, [0]]-bin_cam.color_intr[0, 2], observed_z/bin_cam.color_intr[0, 0])
        observed_y = np.multiply(
            observed_pix[:, [1]]-bin_cam.color_intr[1, 2], observed_z/bin_cam.color_intr[1, 1])
        new_observed_pts = np.concatenate(
            (observed_x, observed_y, observed_z), axis=1)

        # Estimate rigid transform between measured points and new observed points
        R, t = get_rigid_transform(np.asarray(
            measured_pts), np.asarray(new_observed_pts))
        t.shape = (3, 1)
        world2camera = np.concatenate(
            (np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])), axis=0)

        # Compute rigid transform error
        registered_pts = np.dot(R, np.transpose(
            measured_pts)) + np.tile(t, (1, measured_pts.shape[0]))
        error = np.transpose(registered_pts) - new_observed_pts
        error = np.sum(np.multiply(error, error))
        rmse = np.sqrt(error/measured_pts.shape[0])
        return rmse

    # Optimize z scale w.r.t. rigid transform error
    print('Calibrating...')
    z_scale_init = 1
    optim_result = optimize.minimize(
        get_rigid_transform_error, np.asarray(z_scale_init), method='Nelder-Mead')
    camera_depth_offset = optim_result.x

    # Save camera optimized offset and camera pose
    print('Saving calibration files...')
    np.savetxt('real_world/camera_depth_scale.txt', camera_depth_offset, delimiter=' ')
    get_rigid_transform_error(camera_depth_offset)
    camera_pose = np.linalg.inv(world2camera)
    np.savetxt('real_world/camera_pose.txt', camera_pose, delimiter=' ')
    print('Done. Please restart main script.')
