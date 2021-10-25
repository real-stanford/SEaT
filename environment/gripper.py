import pybullet as p
import numpy as np
from environment.utils import get_body_colors, set_visible
from pathlib import Path


class Gripper:
    def __init__(self, robot_body_id, robot_ee_link_index, env):
        self.robot_body_id = robot_body_id
        self.robot_ee_link_index =robot_ee_link_index 
        self.env = env
        self.visual_data = None

    def get_robot_grasp_pose(self, grasp_position, grasp_angle):
        grasp_ori = p.getQuaternionFromEuler(
            [np.pi, 0, grasp_angle])
        return (grasp_position, grasp_ori)

    def prepare_grasp(self):
        # Do nothing
        pass

    def grasp(self):
        pass

    def check_grasp_success(self):
        return False

    def release(self):
        pass

    def set_visible(self, visible=True):
        set_visible(self.visual_data, visible)

class Robotiq2FGripper(Gripper):
    def __init__(self, robot_body_id, robot_ee_link_index, env):
        super().__init__(robot_body_id, robot_ee_link_index, env)

        assets_path = Path(__file__).parent.parent / "assets/"
        self._tool_tip_to_ee_joint = np.array([0, 0, 0.15])
        self.gripper_body_id = p.loadURDF(
            str(assets_path / "gripper/robotiq_2f_85.urdf"),
            basePosition=[0.5, 0.1, 0.2],
            baseOrientation=p.getQuaternionFromEuler([np.pi, 0, 0]))
        p.createConstraint(
            self.robot_body_id, self.robot_ee_link_index,
            self.gripper_body_id, 0,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, -0.05],
            childFrameOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]))

        # Set friction coefficients for gripper fingers
        for i in range(p.getNumJoints(self.gripper_body_id)):
            p.changeDynamics(
                self.gripper_body_id, i,
                lateralFriction=1.0, spinningFriction=1.0,
                rollingFriction=0.0001, frictionAnchor=True)
        self.env.register_callback(self.ensureConstraints, ())
        env.step_simulation(100)

        self.num_joints = p.getNumJoints(self.gripper_body_id)
        self.visual_data = {
            self.gripper_body_id: get_body_colors(self.gripper_body_id),
        }

    def ensureConstraints(self):
        if self.gripper_body_id is None:
            return
        # Constraints
        gripper_joint_positions = np.array([
            p.getJointState(self.gripper_body_id, i)[0]
            for i in range(self.num_joints)])
        p.setJointMotorControlArray(
            self.gripper_body_id, [6, 3, 8, 5, 10], p.POSITION_CONTROL,
            [
                gripper_joint_positions[1], -gripper_joint_positions[1], 
                -gripper_joint_positions[1], gripper_joint_positions[1],
                gripper_joint_positions[1]
            ],
            positionGains=np.ones(5)
        )

    def check_grasp_success(self):
        return p.getJointState(self.gripper_body_id, 1)[0] < 0.833

    def open_gripper(self):
        # Open gripper
        p.setJointMotorControl2(
            self.gripper_body_id, 1, p.VELOCITY_CONTROL, targetVelocity=-5, force=10000)
        self.env.step_simulation(4e2)

    def prepare_grasp(self):
        self.open_gripper()

    def grasp(self, obj_ids=None):
        p.setJointMotorControl2(
            self.gripper_body_id, 1, p.VELOCITY_CONTROL, targetVelocity=5, force=10000)
        self.env.step_simulation(4e2)
    
    def release(self):
        self.open_gripper()

    def get_robot_grasp_pose(self, grasp_position, grasp_angle):
        grasp_position = grasp_position + self._tool_tip_to_ee_joint
        gripper_orientation = p.getQuaternionFromEuler(
            [np.pi, 0, grasp_angle])
        return (grasp_position, gripper_orientation)


class SuctionGripper(Gripper):
    # Reference: Code is borrowed from https://github.com/google-research/ravens
    # Credits: Andy Zeng https://andyzeng.github.io/
    def __init__(self, robot_body_id, robot_ee_link_index, env):
        super().__init__(robot_body_id, robot_ee_link_index, env)
        base = p.loadURDF(
            'assets/ur5/suction/suction-base.urdf',
            (0.487, 0.109, 0.438),
            p.getQuaternionFromEuler((np.pi, 0, 0)))
        p.createConstraint(
            parentBodyUniqueId=robot_body_id,
            parentLinkIndex=robot_ee_link_index,
            childBodyUniqueId=base,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0.01))

        self.gripper_body_id = p.loadURDF(
            'assets/ur5/suction/suction-head.urdf',
            (0.487, 0.109, 0.347),
            p.getQuaternionFromEuler((np.pi, 0, 0)))
        constraint_id = p.createConstraint(
            parentBodyUniqueId=robot_body_id,
            parentLinkIndex=robot_ee_link_index,
            childBodyUniqueId=self.gripper_body_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.08))
        p.changeConstraint(constraint_id, maxForce=50)
        env.step_simulation(100)

        self.activated = False
        self.contact_constraint = None

        self.visual_data = {
            base: get_body_colors(base),
            self.gripper_body_id: get_body_colors(self.gripper_body_id)
        }

    def get_robot_grasp_pose(self, grasp_position, grasp_angle):
        grasp_position = grasp_position + np.array([0, 0, 0.1])
        gripper_orientation = p.getQuaternionFromEuler(
            [np.pi, 0, grasp_angle])
        return (grasp_position, gripper_orientation)

    def grasp(self, obj_ids):
        if self.activated:
            return

        points = p.getContactPoints(bodyA=self.gripper_body_id, linkIndexA=0)
        if points:
            # Handle contact between suction with a rigid object.
            obj_id, contact_link = points[-1][2], points[-1][4]
            if obj_ids is None or obj_id in obj_ids:
                body_pose = p.getLinkState(self.gripper_body_id, 0)
                obj_pose = p.getBasePositionAndOrientation(obj_id)
                world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                obj_to_body = p.multiplyTransforms(world_to_body[0],
                                                    world_to_body[1],
                                                    obj_pose[0], obj_pose[1])
                self.contact_constraint = p.createConstraint(
                    parentBodyUniqueId=self.gripper_body_id,
                    parentLinkIndex=0,
                    childBodyUniqueId=obj_id,
                    childLinkIndex=contact_link,
                    jointType=p.JOINT_FIXED,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=obj_to_body[0],
                    parentFrameOrientation=obj_to_body[1],
                    childFramePosition=(0, 0, 0),
                    childFrameOrientation=(0, 0, 0))
                self.activated = True

    def check_grasp_success(self):
        return self.activated

    def release(self):
        if not self.activated or self.contact_constraint is None:
            self.activated = False
            return

        try:
            p.removeConstraint(self.contact_constraint)
            self.contact_constraint = None
        except:  # pylint: disable=bare-except
            pass