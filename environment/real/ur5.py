import socket
import struct
from time import time, sleep
import numpy as np
import threading
import sys
import signal
import os
import math
from urx import Robot
import math3d as m3d

def get_robot_ip():
    robot_ip_key = "ROBOT_IP"
    if robot_ip_key not in os.environ:
        print("Environment variable ROBOT_IP is not found.")
        raise ValueError("ROBOT_IP not found")
    return os.environ[robot_ip_key]

deg_rad = lambda deg: (deg / 180) * np.pi

class UR5_URX(Robot):
    """
    ur5 controller for this project based on urx controller
    """
    def __init__(self, j_acc=0.1, j_vel=0.1, tool_offset=np.ones(3)) -> None:
        host = get_robot_ip()
        super().__init__(host)
        self.j_acc = j_acc
        self.j_vel = j_vel
        self.__home_j_config = np.asarray(
            [deg_rad(x) for x in [90, -135, 90, -45, -90, 0]])
        print("UR5:- setting tool offset to: ", tool_offset)
        self.set_tcp(tool_offset)

    def homej(self):
        self.movej(self.__home_j_config, self.j_acc, self.j_vel)
    
    def ensure_non_none_acc_vel(self, acc: float, vel: float):
        if acc is None:
            acc = self.j_acc
        if vel is None:
            vel = self.j_vel
        return acc, vel 

    def set_pos_derived(self, vect, acc: float = None, vel: float = None, wait: bool = True, threshold: float = None):
        acc, vel = self.ensure_non_none_acc_vel(acc, vel)
        self.set_pos(vect, acc, vel, wait, threshold)

    def set_pose_derived(self, trans, acc: float = None, vel: float = None, wait: bool = True, command: str = "movel", threshold: float = None):
        acc, vel = self.ensure_non_none_acc_vel(acc, vel)
        # Clip the yaw between -2pi / 3 to 2pi / 3
        # This is required to avoid collision with XYZ gripper suction tube
        trans = np.copy(trans)

        if not isinstance(trans, m3d.Transform):
            trans = m3d.Transform(trans)

        self.set_pose(trans, acc, vel, wait, command, threshold)

    def open_gripper(self):
        self.set_digital_out(6, False)
        sleep(1)

    def close_gripper(self):
        self.set_digital_out(6, True)
        sleep(1)


# Connect to UR5 robot on a real platform
class UR5(object):
    def __init__(self, j_acc=0.1, j_vel=0.1, tool_offset=[0, 0, 0.25, 0, 0, 0]):
        # Set default robot joint acceleration (rad/s^2) and joint velocity (rad/s)
        self.__j_acc = j_acc
        self.__j_vel = j_vel

        # Connect to robot
        self.__tcp_ip = get_robot_ip()
        self.__tcp_port = 30002
        self.setup_tcp_sock()

        # Tool offset for gripper
        self.tool_offset = tool_offset
        while True:
            tcp_msg = 'set_tcp(p[%f,%f,%f,%f,%f,%f])\n' % tuple(self.tool_offset)
            self.__tcp_sock.send(str.encode(tcp_msg))
            print("UR5: set tool offset to ", tool_offset)
            # This while loop is a hack. Please don't judge me!
            break

        self._suction_seal_threshold = 2.5

        # Set home joint configuration
        self.__home_j_config = np.asarray(
            [-270.0, -110.0, 90.0, -90.0, -90.0, 0.0]
        ) * np.pi/180.0

        # Set joint position and tool pose tolerance (epsilon) for blocking calls
        # self.__j_pos_eps = 0.01  # joints
        self.__j_pos_eps = 0.05  # joints
        self.__tool_pose_eps = [0.01, 0.01,
                                0.01, 0.01, 0.01, 0.01]  # tool pose

        # Define Denavit-Hartenberg parameters for UR5
        self._ur5_kinematics_d = np.array(
            [0.089159, 0., 0., 0.10915, 0.09465, 0.0823])
        self._ur5_kinematics_a = np.array([0., -0.42500, -0.39225, 0., 0., 0.])


        # Adding signal handlers to stop robot in case following interrupts occur while
        # the robot is still in motion
        print("UR5: registering signal handlers for ctrl+c, ctrl+z")
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTSTP, self.signal_handler)

    def setup_tcp_sock(self):
        self.__tcp_sock = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self.__tcp_sock.connect((self.__tcp_ip, self.__tcp_port))
        self.__tcp_sock.settimeout(3)

    def __get_state_data(self):
        self.setup_tcp_sock()
        state_data = None
        state_updated = False
        try:
            max_tcp_msg_size = 2048
            since = time()
            while True:
                if time() - since < 3:
                    message_size_bytes = bytearray(self.__tcp_sock.recv(4))
                    if len(message_size_bytes) < 4:
                        # Unpacking into int requires 4 bytes
                        continue
                        
                    message_size = struct.unpack("!i", message_size_bytes)[0]
                    # This is hacky but it can work for multiple versions
                    if message_size <= 55 or message_size >= max_tcp_msg_size:
                        continue
                    else:
                        state_data = self.__tcp_sock.recv(message_size-4)
                    if message_size < max_tcp_msg_size and message_size-4 == len(state_data):
                        self.__state_data = state_data
                        state_updated = True
                        break
                else:
                    print(
                        'Timeout: retrieving TCP message exceeded 3 seconds. Restarting connection.')
                    break
        except socket.timeout:
            print("Timeout exception")
        return state_updated, state_data
        
    # Parse TCP message describing robot state (from primary client)
    def parse_state_data(self, state_data, req_info):
        # Helper function to skip to specific package byte index in TCP message
        def skip_to_package_index(state_data, pkg_type):
            _ = struct.unpack('!B', state_data[0: 1])[0]
            byte_index = 1
            while byte_index < len(state_data):
                package_size = struct.unpack(
                    "!i", state_data[byte_index:(byte_index+4)])[0]
                byte_index += 4
                package_index = int(struct.unpack(
                    '!B', state_data[(byte_index+0):(byte_index+1)])[0])
                if package_index == pkg_type:
                    byte_index += 1
                    break
                byte_index += package_size - 4
            return byte_index

        # Define functions to parse TCP message for each type of requested information
        def parse_timestamp(state_data):
            byte_index = skip_to_package_index(state_data, pkg_type=0)
            timestamp = struct.unpack(
                '!Q', state_data[(byte_index+0):(byte_index+8)])[0]
            return timestamp

        def parse_actual_j_pos(state_data):
            byte_index = skip_to_package_index(state_data, pkg_type=1)
            actual_j_pos = [0, 0, 0, 0, 0, 0]
            for i in range(6):
                actual_j_pos[i] = struct.unpack(
                    '!d', state_data[(byte_index+0):(byte_index+8)])[0]
                byte_index += 41
            return actual_j_pos

        def parse_actual_j_vel(state_data):
            byte_index = skip_to_package_index(state_data, pkg_type=1)+16
            actual_j_vel = [0, 0, 0, 0, 0, 0]
            for i in range(6):
                actual_j_vel[i] = struct.unpack(
                    '!d', state_data[(byte_index+0):(byte_index+8)])[0]
                byte_index += 41
            return actual_j_vel

        def parse_actual_j_currents(state_data):
            byte_index = skip_to_package_index(state_data, pkg_type=1)+24
            actual_j_currents = [0, 0, 0, 0, 0, 0]
            for i in range(6):
                actual_j_currents[i] = struct.unpack(
                    '!f', state_data[(byte_index+0):(byte_index+4)])[0]
                byte_index += 41
            return actual_j_currents

        def parse_actual_tool_pose(state_data):
            byte_index = skip_to_package_index(state_data, pkg_type=4)
            actual_tool_pose = [0, 0, 0, 0, 0, 0]
            for i in range(6):
                actual_tool_pose[i] = struct.unpack(
                    '!d', state_data[(byte_index+0):(byte_index+8)])[0]
                byte_index += 8
            return actual_tool_pose

        def parse_tool_analog_input2(state_data):
            byte_index = skip_to_package_index(state_data, pkg_type=2)+2
            tool_analog_input2 = struct.unpack(
                '!d', state_data[(byte_index+0):(byte_index+8)])[0]
            return tool_analog_input2

        def parse_analog_input1(state_data):
            byte_index = skip_to_package_index(state_data, pkg_type=3)+14
            analog_input1 = struct.unpack(
                '!d', state_data[(byte_index+0):(byte_index+8)])[0]
            return analog_input1

        # Map requested info to parsing function and sub-package type
        parse_func = {
            'timestamp': parse_timestamp,
            'actual_j_pos': parse_actual_j_pos,
            'actual_j_vel': parse_actual_j_vel,
            'actual_j_currents': parse_actual_j_currents,
            'actual_tool_pose': parse_actual_tool_pose,
            'tool_analog_input2': parse_tool_analog_input2,
            'analog_input1': parse_analog_input1
        }
        return parse_func[req_info](state_data)


    # Move joints to specified positions or move tool to specified pose
    def movej(self, use_pos, params, blocking=False, j_acc=None, j_vel=None):
        # Apply default joint speeds
        if j_acc is None:
            j_acc = self.__j_acc
        if j_vel is None:
            j_vel = self.__j_vel

        # Move robot
        tcp_msg = "def process():\n"
        tcp_msg += f" stopj({j_acc})\n"
        tcp_msg += f" movel({'p' if use_pos else ''}[{params[0]},{params[1]},{params[2]},{params[3]},{params[4]},{params[5]}],a={j_acc},v={j_vel},t=0.0,r=0.0)\n"
        tcp_msg += "end\n"
        self.__tcp_sock.send(str.encode(tcp_msg))

        params = np.array(params)
        while blocking:
            state_updated = False
            while not state_updated:
                state_updated, state_data = self.__get_state_data()
                sleep(0.01)
            
            actual_j_vel = self.parse_state_data(state_data, 'actual_j_vel')
            if use_pos:
                actual_tool_pose = self.parse_state_data(
                    state_data, 'actual_tool_pose')
                # Handle repeat axis angle rotations
                actual_tool_pose_mirror = np.copy(actual_tool_pose)
                actual_tool_pose_mirror[3:6] = -actual_tool_pose_mirror[3:6]
                delta1 = np.abs(actual_tool_pose - params)
                delta2 = np.abs(actual_tool_pose_mirror - params)
                if (np.all(delta1 < self.__tool_pose_eps) or np.all(delta2 < self.__tool_pose_eps))\
                    and np.sum(actual_j_vel) < 0.01:
                    break
                else:
                    sleep(0.01)
            else:
                actual_j_pos = self.parse_state_data(state_data, 'actual_j_pos')
                if np.all(np.abs(actual_j_pos-params) < self.__j_pos_eps) and np.sum(actual_j_vel) < 0.01:
                    break
                else:
                    sleep(0.01)

    # Move joints to home joint configuration
    def homej(self, blocking=False):
        self.movej(use_pos=False, params=self.__home_j_config, blocking=blocking)

    def close_gripper(self, blocking=False):
        tcp_msg = "set_digital_out(6,True)\n"
        self.__tcp_sock.send(str.encode(tcp_msg))
        if blocking:
            sleep(0.5)
        return True  # gripper_closed

    def open_gripper(self, blocking=False):
        tcp_msg = "set_digital_out(6,False)\n"
        self.__tcp_sock.send(str.encode(tcp_msg))
        if blocking:
            sleep(0.5)

    # Check if something is in between gripper fingers by measuring grasp width
    def check_grasp(self):
        state_data = self.__state_data
        analog_input1 = self.parse_state_data(state_data, 'analog_input1')

        # Find peak in analog input
        timeout_t0 = time()
        while True:
            state_data = self.__state_data
            new_analog_input1 = self.parse_state_data(
                state_data, 'analog_input1')
            timeout_t1 = time()
            if (
                new_analog_input1 > 2.0 and
                abs(new_analog_input1 - analog_input1) > 0.0 and
                abs(new_analog_input1 - analog_input1) < 0.1
            ) or timeout_t1 - timeout_t0 > 5:
                print(analog_input1)
                return analog_input1 > self._suction_seal_threshold
            analog_input1 = new_analog_input1

    def signal_handler(self, sig, frame):
        # Send stop joints signal to robot in case of interrupt signals and gracefully exit
        tcp_msg = 'def process():\n'
        tcp_msg += ' stopj(%f)\n' % (self.__j_acc)
        tcp_msg += 'end\n'
        self.__tcp_sock.send(str.encode(tcp_msg))
        sys.exit(0)

