from omegaconf import base
import pybullet as p
import numpy as np
import time
from pathlib import Path
from environment.gripper import SuctionGripper
from environment.utils import get_body_colors, set_visible


class UR5:
    def __init__(self, env, basePosition=[0, 0, 0], gripper_class=SuctionGripper):
        self.env = env
        assets_path = Path(__file__).parent.parent / "assets/"
        self.robot_body_id = p.loadURDF(
            str(assets_path / "ur5/ur5.urdf"),
            [basePosition[0], basePosition[1], basePosition[2] + 0.4], p.getQuaternionFromEuler([0, 0, 0]))
        self.mount_body_id = p.loadURDF(
            str(assets_path / "ur5/mount.urdf"),
            [basePosition[0], basePosition[1], basePosition[2] + 0.2], p.getQuaternionFromEuler([0, 0, 0]))

        robot_joint_info = [p.getJointInfo(self.robot_body_id, i) for i in range(
            p.getNumJoints(self.robot_body_id))]
        self._robot_joint_indices = [
            x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]

        self._joint_epsilon = 1e-2

        self.robot_home_joint_config = [
            -np.pi, -np.pi/2, 0, -np.pi/2, -np.pi/2, 0]
        self.move_joints(self.robot_home_joint_config, speed=1.0)

        self.visual_data = {
            self.robot_body_id: get_body_colors(self.robot_body_id),
            self.mount_body_id: get_body_colors(self.mount_body_id),
        }

        # Load gripper
        self.robot_end_effector_link_index = 9
        self.gripper = gripper_class(
                self.robot_body_id, self.robot_end_effector_link_index, self.env)

    def move_joints(self, target_joint_state, speed=0.03):
        """
            Move robot arm to specified joint configuration by appropriate motor control
        """
        assert len(self._robot_joint_indices) == len(target_joint_state)
        p.setJointMotorControlArray(
            self.robot_body_id, self._robot_joint_indices,
            p.POSITION_CONTROL, target_joint_state,
            positionGains=speed * np.ones(len(self._robot_joint_indices))
        )

        timeout_t0 = time.time()
        while True:
            # Keep moving until joints reach the target configuration
            current_joint_state = [
                p.getJointState(self.robot_body_id, i)[0]
                for i in self._robot_joint_indices
            ]
            if all([
                np.abs(
                    current_joint_state[i] - target_joint_state[i]) < self._joint_epsilon
                for i in range(len(self._robot_joint_indices))
            ]):
                break
            if time.time()-timeout_t0 > 5:
                print(
                    "Timeout: robot is taking longer than 5s to reach the target joint state. Skipping...")
                p.setJointMotorControlArray(
                    self.robot_body_id, self._robot_joint_indices,
                    p.POSITION_CONTROL, self.robot_home_joint_config,
                    positionGains=np.ones(len(self._robot_joint_indices))
                )
                break
            self.env.step_simulation(1)

    def set_joint_positions(self, values):
        assert len(self._robot_joint_indices) == len(values)
        for joint, value in zip(self._robot_joint_indices, values):
            p.resetJointState(self.robot_body_id, joint, value)

    def move_tool(self, position, orientation, speed=0.03):
        """
            Move robot tool (end-effector) to a specified pose
            @param position: Target position of the end-effector link
            @param orientation: Target orientation of the end-effector link
        """
        target_joint_state = np.zeros(
            (6,))  # this should contain appropriate joint angle values
        target_joint_state = np.array(
            p.calculateInverseKinematics(self.robot_body_id, self.robot_end_effector_link_index,
                                         position, orientation, residualThreshold=1e-4, maxNumIterations=100)
        )
        self.move_joints(target_joint_state)

    def robot_go_home(self, speed=0.1):
        self.move_joints(self.robot_home_joint_config, speed)

    def execute_pre_grasp(self, pos, ori):
        pre_grasp_position_over_bin = pos+np.array([0, 0, 0.3])
        pre_grasp_position_over_object = pos+np.array([0, 0, 0.1])
        self.move_tool(pre_grasp_position_over_bin,
                       ori, speed=0.01)
        self.move_tool(pre_grasp_position_over_object,
                    ori, speed=0.01)
    
    def execute_post_grasp(self, pos, ori):
        post_grasp_position_over_bin = pos+np.array([0, 0, 0.3])
        self.move_tool(post_grasp_position_over_bin,
                       ori, speed=0.01)
        self.robot_go_home(speed=0.01)
    
    def execute_grasp(self, pos, angle=0, obj_ids=None):
        """
            Execute grasp sequence
            @param: pos: 3d position of place where the gripper jaws will be closed
            @param: angle: angle of gripper before executing grasp from positive x axis in radians 
        """
        # Adjust pos to account for end-effector length
        pos, ori = self.gripper.get_robot_grasp_pose(pos, angle)
        self.gripper.prepare_grasp()
        self.execute_pre_grasp(pos, ori)
        self.move_tool(pos, ori, speed=0.01)
        self.gripper.grasp(obj_ids)
        self.execute_post_grasp(pos, ori)
        return self.gripper.check_grasp_success()
    
    def execute_place(self, pos, angle=0):
        pos, ori = self.gripper.get_robot_grasp_pose(pos, angle)
        self.execute_pre_grasp(pos, ori)
        self.move_tool(pos, ori, speed=0.01)
        self.gripper.release()
        self.execute_post_grasp(pos, ori)

    def execute_pick_place(self, pick_pose, place_pose, obj_ids):
        if not self.execute_grasp(pick_pose[0], p.getEulerFromQuaternion(pick_pose[1])[2], obj_ids):
            return False
        self.execute_place(place_pose[0], p.getEulerFromQuaternion(place_pose[1])[2])
        return True 

    def set_visible(self, visible=True):
        set_visible(self.visual_data, visible=visible)
        self.gripper.set_visible(visible=visible)
