import pybullet as p
import pybullet_data
import time
from pathlib import Path
from imageio import get_writer
from pygifsicle import optimize
from .camera import SimCameraBase
from utils.pointcloud import PointCloud
import numpy as np
from utils.ravenutils import POSETYPE
from environment.utils import get_scene_volume
from collections import defaultdict
from utils.ravenutils import get_fused_heightmap
from PIL import Image

class BaseEnv:
    """
    BaseEnv: Implements two tote UR5 simulation environment with obstacles for grasping 
        and manipulation
    """
    def __init__(self, gui=False):
        # TODO: NEed to make this class singleton
        # - Existence of too different environments simultaneously on same machine may cause issues

        # load environment
        self.gui = gui
        if not p.isConnected():
            p.connect(p.GUI if self.gui else p.DIRECT)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.step_callbacks = list() 
        self.step_callbacks_args = list()
        self.cameras = list()

    def register_callback(self, callback, args):
        """
            @param step_callbacks: list of callback functions to call after every simulation step
        """
        self.step_callbacks.append(callback)
        self.step_callbacks_args.append(args)

    def step_simulation(self, num_steps, sleep=False):
        for _ in range(int(num_steps)):
            p.stepSimulation()
            for callback, args in zip(self.step_callbacks, self.step_callbacks_args):
                callback(*args)
            if sleep and self.gui:
                time.sleep(1e-3)

    def get_scene_volume(self, cameras, view_bounds, return_first_image=False):
        return get_scene_volume(cameras, view_bounds, self.voxel_size, return_first_image) 
    
    def get_scene_gif(self, gif_path:Path, cameras=None):
        if cameras is None:
            cameras = self.cameras

        images = [camera.get_image()[0] for camera in cameras]
        if len(images) > 0:
            with get_writer(gif_path, mode="I") as writer:
                for image in images:
                    writer.append_data(image)
            optimize(str(gif_path))
    
    def get_scene_pcl(self, camera:SimCameraBase):
        # scene pt cloud
        rgb, d, _ = camera.get_image()
        pcl = PointCloud(rgb, d, camera.intrinsics)  # Green
        pcl.make_pointcloud(camera.pose_matrix)
        return pcl

    # Crop the region around the ground truth location.
    @staticmethod
    def get_scene_cmap_hmap(cameras, bounds, pix_size):
        obs = defaultdict(lambda: list())
        configs = list()
        for i, cam in enumerate(cameras):
            rgb, d, _ = cam.get_image()
            # Image.fromarray(rgb.astype(np.uint8)).save(f'{i}.jpg')
            obs["color"].append(rgb)
            obs["depth"].append(d)
            configs.append(cam.get_config())
        return get_fused_heightmap(obs, configs, bounds, pix_size)