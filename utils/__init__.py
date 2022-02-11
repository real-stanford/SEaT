import numpy as np
import os
from datetime import datetime
import torch
import ray
from omegaconf import DictConfig, OmegaConf, open_dict
from pathlib import Path
import random
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import rotate
from utils.ravenutils import np_unknown_cat
import pybullet as p
from scipy.spatial.transform import Rotation
import sys
from shutil import rmtree

def rand_from_range(r):
    return random.randint(-r+1, r-1)

def rand_from_low_high(l, h):
    r = random.randint(l, h)
    if random.random() > 0.5:
        r *= -1
    return r

def center_crop(vol, center, shape, tensor=True):
    """center crop a tsdf volume"""
    # Pad p1_vol for cropping
    half_vol_shape = np.array(shape) // 2
    center = np.clip(center, np.zeros(3), np.array(vol.shape)-1).astype(int)
    if tensor:
        padding = (
            half_vol_shape[2], half_vol_shape[2], 
            half_vol_shape[1], half_vol_shape[1], 
            half_vol_shape[0], half_vol_shape[0])
        vol_pad = F.pad(vol, padding, "constant", 1)
    else: # numpy
        pad_width=(
            (half_vol_shape[0], half_vol_shape[0]),
            (half_vol_shape[1], half_vol_shape[1]),
            (half_vol_shape[2], half_vol_shape[2]))
        vol_pad = np.pad(vol, pad_width=pad_width, mode="constant", constant_values=1)
    center_shifted = center + half_vol_shape
    # Crop the volume around the center
    coords_min, coords_max = center_shifted - half_vol_shape, center_shifted + half_vol_shape
    vol_cropped = vol_pad[
        coords_min[0]: coords_max[0],
        coords_min[1]: coords_max[1],
        coords_min[2]: coords_max[2],
    ]
    return vol_cropped

def calcMIoU(targs, preds):
    """
        Calculate intersection over union given two tsdf volumes.
        Value <=0 means presence of volume.
        targ: (B, 1, H, W, D)
        pred: (B, 1, H, W, D)
    """
    batch_size = targs.shape[0]
    total_IoU = 0
    for ind in range(batch_size):
        total_IoU += calcIoU(targs[ind, 0], preds[ind, 0])
    return total_IoU / batch_size

def calcIoU(targ, pred):
    """
        Calculate intersection over union given two tsdf volumes.
        Value <=0 means presence of volume.
        targ: (H, W, D)
        pred: (H, W, D)
    """
    if torch.is_tensor(targ):
        targ = targ.detach().cpu().numpy()
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    intersection = np.sum(np.logical_and(targ<=0, pred<=0))
    union = np.sum(np.logical_or(targ<=0, pred<=0))
    if union == 0:
        return 0
    IoU = intersection/union
    return IoU

def depth_to_point_cloud(intrinsics, depth_image):
    """Back project a depth image to a point cloud.
        Note: point clouds are unordered, so any permutation of points in the list is acceptable.
        Note: Only output those points whose depth > 0.
    Args:
        intrinsics (numpy.array [3, 3]): given as [[fu, 0, u0], [0, fv, v0], [0, 0, 1]]
        depth_image (numpy.array [h, w]): each entry is a z depth value.
    Returns:
        numpy.array [n, 3]: each row represents a different valid 3D point.
    """
    u0 = intrinsics[0, 2]
    v0 = intrinsics[1, 2]
    fu = intrinsics[0, 0]
    fv = intrinsics[1, 1]

    V = np.empty_like(depth_image)
    for i in range(depth_image.shape[0]):
        V[i, :] = i
    U = np.empty_like(depth_image)
    for j in range(depth_image.shape[1]):
        U[:, j] = j

    X = (U - u0) * (depth_image) / fu
    Y = (V - v0) * depth_image / fv
    point_cloud = np.stack((X, Y, depth_image), axis=-1)
    point_cloud = point_cloud.reshape((depth_image.shape[0] * depth_image.shape[1], 3))
    return point_cloud

def transform_point3s(t, ps):
    '''
    In:
        t: Numpy array [4x4] to represent a transform
        ps: point3s represented as a numpy array [Nx3], where each row is a point.
    Out:
        Transformed point3s as a numpy array [Nx3].
    Purpose:
        Transfrom point from one space to another.
    '''
    if len(ps.shape) != 2 or ps.shape[1] != 3:
        raise ValueError('Invalid input points p')

    # convert to homogeneous
    ps_homogeneous = np.hstack([ps, np.ones((len(ps), 1), dtype=np.float32)])
    ps_transformed = np.dot(t, ps_homogeneous.T).T

    return ps_transformed[:, :3]


def change_extension(path:str, new_extension:str):
    if not os.path.isfile(path):
        raise Exception(f"expected file, received directory - {path}")
    if not new_extension.startswith("."):
        raise Exception(
            f"extension name should start with '.'. Received {new_extension}")
    dir_path = os.path.dirname(path)
    file_name = os.path.basename(path) 
    return os.path.join(
        dir_path,
        f"{file_name.split('.')[0]}{new_extension}"
    )

def get_data_dump_stamp():
    return datetime.today().strftime("%Y-%m-%d--%H-%M-%S")

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')

def get_logs_dir(prefix):
    stamp = get_data_dump_stamp()
    logs_dir = f"logs/mask_rcnn/{prefix}/{stamp}"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    return logs_dir, stamp

def hdf_overwrite_key_if_exist(grp, key, new_data):
    if key in grp:
        del grp[key]
    grp.create_dataset(key, data=new_data)

def ensure_makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def init_ray(ray_cfg: DictConfig):
    use_ray = True
    if ray_cfg.name == "disabled":
        use_ray = False
    elif ray_cfg.name == "single_node":
        ray.init(num_cpus=ray_cfg.num_cpus, num_gpus=ray_cfg.num_gpus, ignore_reinit_error=True)
    return use_ray

def get_ray_fn(ray_cfg: DictConfig, fn, gpu_frac=None):
    """
    gpu_frac: How much gpu usage each function call will require
    """
    if gpu_frac is None:
        ray_fn = ray.remote(fn) if ray_cfg.name != "disabled" else None
    else:
        print("Using gpu fraction: ", gpu_frac)
        ray_fn = ray.remote(num_gpus=gpu_frac)(fn) if ray_cfg.name != "disabled" else None
    def fn_wrapper(*args, **kwargs):
        if ray_cfg.name != "disabled":
            return ray_fn.remote(*args, **kwargs)
        else:
            return fn(*args, **kwargs)
    return fn_wrapper

def is_file_older(a: Path, b: Path):
    return a.stat().st_ctime > b.stat().st_ctime

def next_loop_iter(dataloader_iter, dataloader):
    try:
        next_batch = next(dataloader_iter)
        return dataloader_iter, next_batch 
    except StopIteration:
        dataloader_iter = iter(dataloader)
        return next_loop_iter(dataloader_iter, dataloader)

def seed_all_int(seed:int):
    # print(f"SEEDING WITH {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def seed_all(cfg:DictConfig):
    if OmegaConf.is_missing(cfg, "seed"):
        cfg.seed = int(random.random() * (2**31))
        print("Seeding with random seed")
    seed = int(cfg.seed)
    return seed_all_int(seed)

def depth_to_pil_image(d):
    return Image.fromarray((((d - d.min()) / (d.max() - d.min())) * 255).astype(np.uint8))


def get_masked_rgb(mask, rgb):
    return np.copy(rgb) * mask[:, :, np.newaxis]

def get_masked_d(mask, d):
    masked_d = np.copy(d) * mask
    masked_d[mask != 1] = 1e5
    return masked_d

def get_tsdf_match(pose0_vol, pose1_vol, pose0_width, pose1_bounds, voxel_size):
    # Negate key
    device = get_device()
    k = torch.tensor(pose1_vol, device=device).unsqueeze(dim=0).unsqueeze(dim=0)
    k *= -1
    k[k > 0] *= 2

    # Rotate query
    n_rotations = 36
    angles = np.linspace(0, 360, n_rotations, endpoint=False)
    pose0_vol_rotations = None
    for _, angle in enumerate(tqdm(angles)):
        r = rotate(pose0_vol, angle, reshape=False)
        pose0_vol_rotations = np_unknown_cat(pose0_vol_rotations, r)

    # Prepare query volume
    pose0_vol_rotations[pose0_vol_rotations > 0] = 0
    pose0_vol_rotations[pose0_vol_rotations < 0] = -1
    q = torch.tensor(pose0_vol_rotations, device=device).unsqueeze(dim=1)
    stride = 1
    score_maps = torch.nn.functional.conv3d(k, q, stride=stride).squeeze()

    m = torch.max(score_maps)
    indices = torch.nonzero(score_maps == m)[0]

    # _, ax = plt.subplots(1,1)
    # ax.imshow(score_maps[indices[0]]) 
    # circ = Circle((indices[2], indices[1]), 1, color="r") 
    # ax.add_patch(circ)
    # ax.set_xlabel(f"m={m} rot={angle} indices: {indices}")
    # plt.show()

    # Ok. What ne
    angle = np.pi * angles[indices[0]] / 180
    position = pose1_bounds[:, 0] + np.array([
        0.5 * pose0_width[0] + stride * indices[1] * voxel_size,
        0.5 * pose0_width[1] + stride * indices[2] * voxel_size,
        0.1  # Z index is fixed. But this can be figured out by the height of the object picked
    ])
    return position, angle


def get_finetuned_place_pos(pose0_vol, pose1_vol, pose1_pos, pose0_width, pose1_bounds, voxel_size):
    # Get the pose1_vol_cropped
    crop_size_kit = np.array([0.4, 0.4])
    hw = np.ceil(crop_size_kit / (2 * voxel_size))
    # I need to figure out indices
    mid_indices = np.ceil((pose1_pos - pose1_bounds[:, 0]) / voxel_size)
    crop_indices = np.empty((3,2))
    for i in range(2):
        crop_indices[i, 0] = max(0, mid_indices[i] - hw[i])
        crop_indices[i, 1] = min(mid_indices[i] + hw[i], pose1_vol.shape[i])
    crop_indices[2, :] = [0, np.ceil(0.1 / voxel_size)]  # Just hardcoding z dimension
    crop_indices = crop_indices.astype(np.int)
    # Now I have crop_indices
    pose1_vol_cropped = pose1_vol[
        crop_indices[0, 0]: crop_indices[0, 1],
        crop_indices[1, 0]: crop_indices[1, 1],
        crop_indices[2, 0]: crop_indices[2, 1],
    ]
    pose1_bounds_cropped = pose1_bounds[:, 0:1] + crop_indices * voxel_size
    pose0_vol_cropped = pose0_vol[:, :, crop_indices[2, 0]: crop_indices[2, 1]]
    return get_tsdf_match(pose0_vol_cropped, pose1_vol_cropped, pose0_width, pose1_bounds_cropped, voxel_size)

def get_pix_size(bounds, img_height):
    return (bounds[0, 1] - bounds[0, 0]) / img_height

def get_pix_from_pos(pos, bounds, pix_size, include_z=False):
    if not include_z:
        return ((pos - bounds[:, 0]) / pix_size)[:2].astype(np.int)
    else:
        return ((pos - bounds[:, 0]) / pix_size).astype(np.int)

def get_pos_from_pix(pix, bounds, pix_size):
    return np.concatenate((bounds[:2, 0] + np.array(pix) * pix_size, [0.1]))

def get_crop(img, pix, crop_size):
    """
    img: np.ndarray [H, W, C]
    Make a crop of (width, height) = crop_size, centered at pix
    The image is padded with 0 value to handle boundaries
    """
    hcs = (crop_size / 2).astype(np.int)
    padding = ((hcs[0], hcs[0]), (hcs[1], hcs[1]))
    if len(img.shape) == 3:
        padding += ((0, 0), )
    img_pad = np.pad(img, padding)
    img_crop = np.zeros((2 * hcs[0],  2 * hcs[1], img.shape[-1]))
    img_crop = img_pad[
        pix[0]: (pix[0] + 2 * hcs[0]),
        pix[1]: (pix[1] + 2 * hcs[1]),
    ]
    return img_crop

def sample_dataset(dataset:Dataset):
    return dataset[random.sample(range(len(dataset)), 1)[0]]


def quaternion_to_v_theta(quaternion: np.ndarray):
    quat_theta = 2 * np.arccos(quaternion[3])
    quat_v = np.array([
        quaternion[0] / np.sin(quat_theta / 2),
        quaternion[1] / np.sin(quat_theta / 2),
        quaternion[2] / np.sin(quat_theta / 2),
    ])
    return quat_v, quat_theta


def v_theta_to_quaternion(quat_v: np.ndarray, quat_theta: float) -> np.ndarray:
    return np.array([
        quat_v[0] * np.sin(quat_theta / 2),
        quat_v[1] * np.sin(quat_theta / 2),
        quat_v[2] * np.sin(quat_theta / 2),
        np.cos(quat_theta / 2),
    ])


def random_direction():
    """
    random direction in 3D sampled unifomrly
    Reference: Based on this great answer:
    @MISC {44691,
        TITLE = {How to find a random axis or unit vector in 3D?},
        AUTHOR = {Jim Belk (https://math.stackexchange.com/users/1726/jim-belk)},
        HOWPUBLISHED = {Mathematics Stack Exchange},
        NOTE = {URL:https://math.stackexchange.com/q/44691 (version: 2011-06-11)},
        EPRINT = {https://math.stackexchange.com/q/44691},
        URL = {https://math.stackexchange.com/q/44691}
    }
    """
    z = random.uniform(-1, 1)
    theta = random.uniform(0, 2*np.pi)
    hyp = np.sqrt(1 - z**2)
    return np.array([hyp * np.cos(theta), hyp * np.sin(theta), z])


def random_quaternion():
    """
    random quaternion sampled unifomrly
    """
    quat_v = random_direction()
    quat_theta = random.uniform(0, 2*np.pi)
    return v_theta_to_quaternion(quat_v, quat_theta)


def get_quaternion_v1_v2(v1, v2):
    """
    Return quaternion required to rotate vector v1 to v2
    Reference: https://stackoverflow.com/a/1171995/5756943
    Pseudo-Code:
    - Use (v1 X v2) to get direction about which to rotate
    - Use <v1, v2> to get amount by which to rotate
    """
    v1 = -1 * np.copy(v1)  # XXX: Not sure why we need to multiply by -1 (but this works)
    quat_v = np.cross(v1, v2)
    quat_v /= np.linalg.norm(quat_v)
    quat_theta = np.arccos(np.dot(v1, v2) / np.linalg.norm(v1))
    return np.array([
        quat_v[0] * np.cos(quat_theta / 2),
        quat_v[1] * np.cos(quat_theta / 2),
        quat_v[2] * np.cos(quat_theta / 2),
        np.sin(quat_theta / 2),
    ])


def distance_quaternion(q1: np.ndarray, q2: np.ndarray) -> float:
    # Using formulae from https://math.stackexchange.com/a/90098/540690
    if len(q1.shape) == 1:  # Single quaternion
        return 1 - (np.dot(q1, q2))**2
    else:  # (n, 4) n quaternions
        return 1 - ((q1 * q2).sum(axis=1))**2


def sample_directions(v1: np.ndarray, num_z: int, num_phi: int, max_pert_theta: float):
    """
    Sample directions in vicinity of direction v1 uniformly
    vicinity: spherical segment defined by max_pert_theta around v1
    granularity: is controlled by num_z, num_phi
    Reference: https://math.stackexchange.com/q/205589
    """
    zs = np.linspace(np.cos(max_pert_theta), 1, num=num_z, endpoint=False)
    phis = np.linspace(0, 2*np.pi, num=num_phi)
    uniform_dirs = list()
    for z in zs:
        for phi in phis:
            hyp = np.sqrt(1 - z**2)
            uniform_dirs.append(np.array([hyp*np.cos(phi), hyp*np.sin(phi), z]))
    # Rotate these directions to be around v1
    quat = get_quaternion_v1_v2(v1, np.array([0, 0, 1]))
    rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3,3)
    for i in range(len(uniform_dirs)):
        uniform_dirs[i] = np.dot(rot_mat, uniform_dirs[i])
    uniform_dirs += [v1]
    return uniform_dirs


def get_rot_mat_from_quat(quat: np.ndarray):
    return np.array(p.getMatrixFromQuaternion(quat)).reshape(3,3)


def sample_quaternions(quaternion: np.ndarray, max_pert_theta: float, max_pert_rot: float, num_z: int, num_phi: int, num_rots: int, return_dirs=False):
    """
    Given a quaternion, sample quaternions uniformly in its vicinity
    - vicinity is defined by max_pert_theta
    - sampling granularity is defined by num_*
    Checkout scripts/visualize_6DoF_angle_sampling.py for visualization
    """
    # Find out the direction of quaternion
    initial_rot = get_rot_mat_from_quat(quaternion)
    v1 = np.dot(initial_rot, np.array([0, 0, 1]))
    v1 /= np.linalg.norm(v1)
    # Get directions in vicinity of v1
    v2s = sample_directions(v1, num_z, num_phi, max_pert_theta)
    quats = None
    
    rot_angles = np.linspace(-max_pert_rot, max_pert_rot, num=num_rots)
    for v2 in v2s:
        # Figure out the quaternion required to rotate object in initial_rot (transformation) direction to v2 direction
        v2 /= np.linalg.norm(v2)
        if np.linalg.norm(v1 - v2) > 1e-8:  # v1 and v2 are same
            quat_v = np.cross(v1, v2)
            quat_v /= np.linalg.norm(quat_v)
            dot_prod = np.clip(np.dot(v1, v2).sum(), -1, 1)
            quat_theta = np.arccos(dot_prod)
            v2_quat = v_theta_to_quaternion(quat_v, quat_theta)
            v2_rot = get_rot_mat_from_quat(v2_quat)
        else:
            v2_rot = np.identity(3)

        # Multiply base quaternion quat1 with rotation (rot radians) quaternion around it's axis (in max_pert_rot vicinity)
        for rot_angle in rot_angles:
            quat2 = v_theta_to_quaternion(v2, rot_angle)
            delta_rot = get_rot_mat_from_quat(quat2)
            final_rot = delta_rot @ v2_rot @ initial_rot
            quat = Rotation.from_matrix(final_rot).as_quat()
            quats = np_unknown_cat(quats, quat)
    if not return_dirs:
        return quats
    else:
        return quats, v2s


def rotate_tsdf(vol, euler, degrees=False):
    """
    rotate 3d Volume by provided euler angles in **radians**
    """
    if not degrees:
        euler_degrees = euler * 180 / np.pi
    vol_rotate = rotate(vol, euler_degrees[0], axes=(1, 2), reshape=True, cval=1) 
    vol_rotate = rotate(vol_rotate, -euler_degrees[1], axes=(2, 0), reshape=True, cval=1) 
    vol_rotate = rotate(vol_rotate, euler_degrees[2], axes=(0, 1), reshape=True, cval=1) 

    # Crop back to original shape around center
    vrs = np.array(vol_rotate.shape)
    vs = np.array(vol.shape)
    if (vrs < vs).any():
        # Sometimes, after rotation, vol_rotate has lesser voxels than the original volume
        # - (TODO: This happens because the original volumes are not aligend at center which probably is another bug to fix later)
        # Add appropriate padding to vrs
        pad_width = np.clip(np.ceil((vs - vrs) / 2), 0, np.infty).astype(np.int)
        padding = (
            (pad_width[0], pad_width[0]),
            (pad_width[1], pad_width[1]),
            (pad_width[2], pad_width[2]),
        )
        vol_rotate = np.pad(vol_rotate, pad_width=padding, mode="constant", constant_values=1)
        vrs = np.array(vol_rotate.shape)
    
    vol_rotate_center = np.ceil(vrs / 2).astype(np.int)
    hw = np.ceil(vs / 2).astype(np.int)
    final_vol = vol_rotate[
        vol_rotate_center[0] - hw[0]: vol_rotate_center[0] + vs[0] - hw[0],
        vol_rotate_center[1] - hw[1]: vol_rotate_center[1] + vs[1] - hw[1],
        vol_rotate_center[2] - hw[2]: vol_rotate_center[2] + vs[2] - hw[2],
    ]
    return final_vol

def get_dataloader(dataset, seed, **kwargs):
    worker_init_fn = lambda worker_id: seed_all_int(seed + worker_id)
    return DataLoader(dataset, worker_init_fn=worker_init_fn, **kwargs)


class Logger(object):
    # Source: https://stackoverflow.com/a/14906787/5756943
    def __init__(self, log_file:Path):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    


def init_logs_dir(cfg: DictConfig, name: str = ""):
    # stamp = get_data_dump_stamp()
    logs_dir = Path(f"logs/{name}/{get_data_dump_stamp()}")
    logs_dir.mkdir(exist_ok=True, parents=True)
    with open_dict(cfg):
        cfg.logs_dir = str(logs_dir)

    # Save config
    cfg_save_path = logs_dir / "cfg.yaml"
    with open(cfg_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    sys.stdout = Logger(logs_dir / "logs.txt")
    sys.stderr = sys.stdout
    print("Logs dir init at path ", logs_dir)
    return logs_dir

def convert_index_to_flat_index(vol_shape: np.ndarray, index: np.ndarray):
    left_prod = 1
    flat_index = 0
    for i in range(len(vol_shape)-1, -1, -1):
       flat_index += left_prod * index[i] 
       left_prod *= vol_shape[i]
    return flat_index


def convert_flat_index_to_index(vol_shape: np.ndarray, flat_index: int):
    index = None
    for i in range(len(vol_shape)-1, -1, -1):
        index = np_unknown_cat(index, flat_index % vol_shape[i])
        flat_index //= vol_shape[i]
    return index[::-1]


def get_random_indices_except(vol_shape, except_index, k):
    """
    Given a volume shape, sample k random indices except except_index
    """
    # Cool. 
    vol_shape = np.array(vol_shape)
    except_index = np.array(except_index)
    total_indices = np.prod(vol_shape)
     
    flat_except_index = convert_index_to_flat_index(vol_shape, except_index)
    flat_sample_indices = random.sample(
        list(range(flat_except_index)) + list(range(flat_except_index + 1, total_indices)),
        min(total_indices - 1, k)) 
    sample_indices = np.array([convert_flat_index_to_index(vol_shape, flat_sample_index) for flat_sample_index in flat_sample_indices])
    return sample_indices

def get_bounds_from_center_pt(orig_center_pt: np.ndarray, vol_shape: np.ndarray, voxel_size: float, bounds: np.ndarray) -> np.ndarray:
    """
    Given a center pt, returns crop bounds (for TSDF volume generation) such that:
        a) Crop bounds will be strictly inside the view bounds
        b) Crop bounds will be centered around center pt if (a) is satisfied doing so, otherwise shifted appropriately to satisfy (a)
        c) Size of crop bounds will be such that when used to generate tsdf using these crop bounds and voxel_size, it will generate volume of shape vol_shape
    """
    hw_coords_left = (vol_shape * voxel_size) / 2
    hw_coords_right = vol_shape * voxel_size - hw_coords_left
    center_pt = np.copy(orig_center_pt)
    for j in range(3):
        if orig_center_pt[j] - hw_coords_left[j] < bounds[j, 0]:
            center_pt[j] = bounds[j, 0] + hw_coords_left[j]
        elif orig_center_pt[j] + hw_coords_right[j] > bounds[j, 1]:
            center_pt[j] = bounds[j, 1] - hw_coords_right[j]
    crop_bounds = np.array([
        [center_pt[0] - hw_coords_left[0], center_pt[0] + hw_coords_right[0]],
        [center_pt[1] - hw_coords_left[1], center_pt[1] + hw_coords_right[1]],
        [center_pt[2] - hw_coords_left[2], center_pt[2] + hw_coords_right[2]]
    ])
    # Due to floating point error, sometimes the crop bounds are such that 
    # when converted to vol shape, they are slightly bigger.
    # To remedy this, we find out such indices and make them slightly shorter to just fit our vol_shape
    edit_indices = np.ceil((crop_bounds[:, 1] - crop_bounds[:, 0]) / voxel_size).copy(order="C").astype(np.int) != vol_shape
    if np.any(edit_indices):
        crop_bounds[edit_indices, 1] -= voxel_size
    return crop_bounds

def get_split_obj_roots(split_file: Path):
    obj_paths = list()
    with open(split_file, "r") as f:
        for obj_path in f.readlines():
            obj_root = Path(obj_path.strip()).parent
            if (obj_root / "obj.obj").exists() and\
                (obj_root / "obj.urdf").exists() and\
                (obj_root / "kit_parts/kit.obj").exists():
                obj_paths.append(obj_root)
    return obj_paths


def get_split_file(train_test:str) -> Path:
    root = Path("dataset")
    if train_test not in ["train", "val", "val_real", "train_sc", "val_sc"]:
        raise NotImplementedError
    return root / f"{train_test}.txt"

def mkdir_fresh(dir_path:Path, ask_first: bool = False):
    if dir_path.exists():
        if ask_first:
            print(f"Removing path: {dir_path}. Please press enter to continue:")
            input()
        rmtree(dir_path)
    dir_path.mkdir(parents=True)
    return dir_path


def pad_crop_to_size(vol, required_shape):
    updated = False
    initial_shape = vol.shape
    i = 0
    for i in range(3):
        if vol.shape[i] < required_shape[i]:
            pad_width = np.zeros((3, 2), dtype=int)
            width = required_shape[i] - vol.shape[i] 
            pad_width[i][0] = np.floor(width / 2).astype(int)
            pad_width[i][0] = width - pad_width[i][0]
            pad_width = tuple(tuple(x) for x in pad_width)
            vol = np.pad(vol, pad_width, constant_values=1)
            updated = True
    if updated:
        print(f"NOTE: updated vol shape from {initial_shape} to {vol.shape}")
    return vol

def reduce_to_shape(vol, required_shape):
    updated = False
    initial_shape = vol.shape
    i = 0
    for i in range(3):
        if vol.shape[i] > required_shape[i]:
            remove_width = vol.shape[i] - required_shape[i]
            remove_width_left = np.floor(remove_width / 2).astype(int)
            remove_width_right = remove_width - remove_width_left
            if i == 0:
                vol = vol[remove_width_left:-remove_width_right, :, :]
            elif i == 1:
                vol = vol[:, remove_width_left:-remove_width_right, :]
            else:
                vol = vol[:, :, remove_width_left:-remove_width_right]
            updated = True
    if updated:
        print(f"NOTE: updated vol shape from {initial_shape} to {vol.shape}")
    return vol
    
def ensure_vol_shape(vol, required_shape):
    vol = pad_crop_to_size(vol, required_shape)
    vol = reduce_to_shape(vol, required_shape)
    return vol

def show_overlay_image(base_img: np.ndarray, overlay_img: np.ndarray, title: str = ""):
    plt.imshow(base_img)
    plt.imshow(overlay_img, alpha=0.7)
    plt.title(title)
    plt.show()


def save_img(img: np.ndarray, path: Path, cmap: str = None) -> Path:
    if cmap is None:
        plt.imsave(path, img)
    else:
        plt.imsave(path, img, cmap=cmap)
    return path