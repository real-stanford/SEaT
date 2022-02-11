from copy import deepcopy
from pathlib import Path
from sys import dont_write_bytecode

from numpy.lib.function_base import append, diff
from numpy.lib.index_tricks import diag_indices
from environment.meshRendererEnv import MeshRendererEnv
from omegaconf import DictConfig
import numpy as np
import hydra
import torch
from utils import (
    get_device,
    seed_all_int,
)
from utils.pointcloud import get_pointcloud_color, write_pointcloud
from utils.tsdfHelper import TSDFHelper, extend_to_bottom, get_single_biggest_cc_single
from real_world.rw_utils import get_intrinsics, get_empty_depth, get_tn_bounds
from learning.srg import SRG
from learning.dataset import ResultDataset
from learning.vol_match_rotate import VolMatchRotate
from learning.vol_match_transport import VolMatchTransport
from baseline.transportnet import Transportnet
from environment.camera import SimCameraYawPitchRoll
import matplotlib.pyplot as plt
from environment.teleRobotEnv import TeleRobotEnv
from environment.camera import SimCameraPosition 
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from PIL import Image
import cv2

C1 = np.array([78, 121, 167 , 255 / 2]) / 255 # blue
C2 = np.array([237, 201, 72, 255 / 2]) / 255 # yellow

output_dir = Path('obj_kit_pairs')

def save_completed_scene(mesh_path, cnt):

    mesh_path = Path(mesh_path)
    folder_path = mesh_path.parents[0]
    obj_path = folder_path / 'obj.obj'
    kit_path = folder_path / 'kit_parts/kit.obj'
    vis_env = MeshRendererEnv(gui=False)
    vis_env.load_mesh(obj_path, [0,-0.05,0], rgba=C2)
    vis_env.load_mesh(kit_path, [0,0.05,0], rgba=C1)

    look_at = np.array([0, 0, 0])
    r, theta, phi = 0.4, 0.7, 0
    eyepos = r *  np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    eyepos = eyepos + look_at
    camera = SimCameraPosition(eyepos, look_at, image_size=(720,720))
    rgb = camera.get_image()[0]
    im = Image.fromarray(rgb)
    im.save(str(output_dir / f'{cnt}.png'))

output_dir.mkdir(exist_ok=True)
file_paths = open('dataset/train.txt').readlines()
for i, path in enumerate(file_paths):
    save_completed_scene(path, i)