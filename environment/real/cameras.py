#!/usr/bin/env python

import socket
import numpy as np
import threading
import time
from pathlib import Path
from matplotlib import pyplot as plt


class RealSense(object):
    def __init__(
        self, tcp_ip='127.0.0.1', tcp_port=50010, im_h=720, im_w=1280, max_depth=3.0,
        cam_pose_path: Path = Path("real_world/camera_pose.txt"),
        cam_depth_scale_path: Path = Path("real_world/camera_depth_scale.txt")
    ):
        self.tcp_ip = tcp_ip
        self.tcp_port = tcp_port
        self.im_h = im_h
        self.im_w = im_w
        self.max_depth = max_depth  # in meters
        self.buffer_size = 10*4 + self.im_h*self.im_w*5  # in bytes

        # Connect to server
        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.connect((self.tcp_ip, self.tcp_port))

        if cam_pose_path is None:
            print("Using identity camera pose")
            self.pose = np.eye(4)
        else:
            print("Using camera pose from path: ", cam_pose_path)
            self.pose = np.loadtxt(cam_pose_path)

        if cam_depth_scale_path is None:
            print("Using unit depth scale")
            self.depth_scale = 1
        else:
            print("Using depth scale from path: ", cam_depth_scale_path)
            self.depth_scale = np.loadtxt(cam_depth_scale_path)
        self.max_depth *= self.depth_scale

        # Fetch data continually
        self._color_im = None
        self._depth_im = None
        self._timestamp = None
        self._color_intr = None
        self._depth_intr = None
        self._depth2color_extr = None
        self.last_update_time = np.NINF
        while self._depth_im is None or self._color_im is None or self._color_intr is None:
            self.get_data() 

        # rgb, d = self.get_camera_data() 
        # plt.imshow(rgb, alpha=0.5)
        # plt.imshow(d, alpha=0.5)
        # plt.show()
        # print("IMPORTANT: Please make sure that the rgb and depth are aligned")

    # Get latest RGB-D frame and intrinsics

    def get_data(self):
        if time.time() - self.last_update_time < 0.001:
            return 
        self.last_update_time = time.time()
        # Ping the server with anything
        self.tcp_sock.send(b'blah')

        # Fetch TCP data:
        #     color camera intrinsics, 9 floats, number of bytes: 9 x 4
        #     depth camera intrinsics, 9 floats, number of bytes: 9 x 4
        #     depth to color camera extrinsics, 16 floats, number of bytes: 16 x 4
        #     depth scale for converting depth from uint16 to float, 1 float, number of bytes: 4
        #     timestamp (in milliseconds since unix epoch), 1 long, number of bytes: 8
        #     depth image, self.im_w x self.im_h uint16, number of bytes: self.im_w x self.im_h x 2
        #     color image, self.im_w x self.im_h x 3 uint8, number of bytes: self.im_w x self.im_h x 3
        data = b''
        while len(data) < (9*4+9*4+16*4+4+8+self.im_h*self.im_w*5):
            data += self.tcp_sock.recv(self.buffer_size)

        # Re-organize TCP data into color and depth frame
        self._color_intr = np.fromstring(
            data[0:(9*4)], np.float32).reshape(3, 3)
        self._depth_intr = np.fromstring(
            data[(9*4):(9*4+9*4)], np.float32).reshape(3, 3)
        self._depth2color_extr = np.fromstring(
            data[(9*4+9*4):(9*4+9*4+16*4)], np.float32).reshape(4, 4)
        depth_scale = np.fromstring(
            data[(9*4+9*4+16*4):(9*4+9*4+16*4+4)], np.float32)[0]
        self._timestamp = np.fromstring(
            data[(9*4+9*4+16*4+4):(9*4+9*4+16*4+4+8)], np.long)[0]
        depth_im = np.fromstring(data[(9*4+9*4+16*4+4+8):(
            (9*4+9*4+16*4+4+8)+self.im_w*self.im_h*2)], np.uint16).reshape(self.im_h, self.im_w)
        self._color_im = np.fromstring(
            data[((9*4+9*4+16*4+4+8)+self.im_w*self.im_h*2):], np.uint8).reshape(self.im_h, self.im_w, 3)
        depth_im = depth_im.astype(float) * 0.0001  # * depth_scale

        # Set invalid depth pixels to zero
        depth_im = depth_im.copy()
        depth_im[depth_im > self.max_depth] = 0.0
        self._depth_im = depth_im * self.depth_scale
        

    @property
    def color_im(self):
        self.get_data()
        return self._color_im

    @property
    def depth_im(self):
        self.get_data()
        return self._depth_im

    @property
    def color_intr(self):
        # Color Intrinsic is not going to change
        return self._color_intr
    
    @property
    def depth_intr(self):
        return self._depth_intr

    def get_avg_depth(self, n):
        ds = np.empty((self.im_h, self.im_w, n), dtype=np.float64)
        for i in range(n):
            ds[:, :, i] = self.depth_im
            time.sleep(0.001)
        ds[ds == 0] = np.nan
        d = np.nanmedian(ds, axis=2)
        d[np.isnan(d)] = self.max_depth
        d[d == 0] = self.max_depth
        d[d > self.max_depth] = self.max_depth
        return d
    
    def get_camera_data(self, avg_depth=False, avg_over_n=50):
        if avg_depth:
            depth_im = self.get_avg_depth(avg_over_n)
        else:
            depth_im = self.depth_im
        return self.color_im, depth_im


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    cam = RealSense(
        tcp_ip='127.0.0.1', tcp_port=50010,
        im_h=720, im_w=1280, max_depth=3.0
    )
    fig, ax = plt.subplots(1, 2)
    while True:
        ax[0].imshow(cam.color_im)
        ax[1].imshow(cam.depth_im)
        plt.pause(0.005)
        plt.cla()
