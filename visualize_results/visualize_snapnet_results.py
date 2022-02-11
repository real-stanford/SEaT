"""
Generate gt data from saved gt_labels.json
Also run evaluation of models along side
"""

from copy import deepcopy
from pathlib import Path
from sys import dont_write_bytecode

from numpy.lib.function_base import append, diff
from numpy.lib.index_tricks import diag_indices
from numpy.lib.type_check import real
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
B = [ 78,121,167] # blue
RED = [255, 87, 89] # red
G = [ 89,169, 79] # green
O = [242,142, 43] # orange
Y = [237,201, 72] # yellow
P = [176,122,161] # purple
PI = [255,157,167] # pink
C = [118,183,178] # cyan
BR = [156,117, 95] # browne
GRAY = [186,176,172]  # gray
COLORS = np.array([RED,G,O,Y,P,PI,C,BR,GRAY]) / 255
alphas = np.ones((COLORS.shape[0], 1)) * 0.5
COLORS = np.concatenate((COLORS, alphas), axis=1)

HEIGHT = 480 # pixels
IMG_SIZE  = (1000, 1000) # pixels, for predictions
NUM_VIEW = 8

def dump_vol(output_dir, prefix, suffix, voxel_size, vol):
    output_dir.mkdir(exist_ok=True, parents=True)
    mesh_path = output_dir / f"{prefix}_{suffix}.obj"
    TSDFHelper.to_mesh(vol, mesh_path, voxel_size)
    return mesh_path

def center_crop_img(path):
    img = cv2.imread(path)
    size = img.shape[:2]
    # convert
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    # Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,50))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    # Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    # crop
    x,y,w,h = cv2.boundingRect(cnt)
    dst = img[y:y+h, x:x+w]
    # scale if too small
    rh, rw = h/size[0], w/size[1]
    if max(rh, rw) < 0.8:
        scale = 0.8 / max(rh, rw)
        new_dim = tuple((np.array([w, h]) * scale).astype(int))
        dst = cv2.resize(dst, new_dim)
    # pad
    pad_size = (np.array(size) - dst.shape[:2])//2
    dst = np.pad(dst, ((pad_size[0],pad_size[0]), (pad_size[1],pad_size[1]), (0, 0)), 
                 'constant', constant_values=255)
    cv2.imwrite(path, dst)

def save_fig(output_dir, name, array, cmap='viridis', center_crop=True, out_height=None):
    output_dir.mkdir(exist_ok=True)
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(array, cmap=cmap)
    output_path = str(output_dir / f"{name}.png")
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    im = Image.open(output_path)
    w, h = im.size
    if out_height is None:
        out_height = HEIGHT 
    im1 = im.resize((int(w * out_height/h), out_height))
    im1.save(output_path)
    if center_crop:
        center_crop_img(output_path)

def save_around_imgs(output_dir, prefix,p1_mesh_path, labels):
    bb_min = -0.12 * np.ones((3,))
    bb_max = 0.12 * np.ones((3,))
    bb = np.vstack((bb_min, bb_max)).T
    tp = (bb[:, 0] + bb[:, 1]) / 2
    dist = np.sqrt(((bb[:, 1] - bb[:, 0])**2).sum()) 

    vis_env = MeshRendererEnv(gui=False)
    vis_env.load_mesh(p1_mesh_path, rgba=C1)
    for i, (p0_mesh_path, p1_pos, p1_ori) in enumerate(labels):
        vis_env.load_mesh(p0_mesh_path, p1_pos, p1_ori, rgba=COLORS[i])
    for i, yaw in enumerate(np.linspace(0, 360, NUM_VIEW)):
        camera = SimCameraYawPitchRoll(tp, dist, yaw, -50, 0, image_size=IMG_SIZE)
        rgb = camera.get_image()[0]
        save_fig(output_dir / f'{i}', f'{prefix}', rgb)
    vis_env.reset()

def get_pos_ori(real_world, pos):
    ori = [0,0,0,1]
    if real_world:
        flipped_pos = np.zeros(3)
        flipped_pos[0] = -pos[0]
        flipped_pos[1] = -pos[1]
        flipped_pos[2] = pos[2]
        ori = [0,0,1,0]
        pos = flipped_pos
    return pos, ori

def save_completed_scene(output_dir, real_world, all_pos, obj_mesh_paths_sc, kit_mesh_path_sc, kit_pos):
    vis_env = MeshRendererEnv(gui=False)
    for i, pos in enumerate(all_pos):
        vis_env.load_mesh(obj_mesh_paths_sc[i], *get_pos_ori(real_world, pos), rgba=COLORS[i])
    vis_env.load_mesh(kit_mesh_path_sc, *get_pos_ori(real_world, kit_pos), rgba=C1)

    if real_world:
        look_at = np.array([0.7, -0.3, 0.2])
        r, theta, phi = 0.5, 0.9, np.pi/180 * (-40)
        eyepos = r *  np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), 0.4])
        eyepos = eyepos + look_at
        camera = SimCameraPosition(eyepos, look_at)
        rgb = camera.get_image()[0]
        rgb_crop = rgb[190:-50, 430:-190]
    else:
        look_at = np.array([0.35, -0.2, 0.2])
        r, theta, phi = 1.3, 0.9, -1
        eyepos = r *  np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        eyepos = eyepos + look_at
        camera = SimCameraPosition(eyepos, look_at)
        rgb = camera.get_image()[0]
        rgb_crop = rgb[190:-50, 320:-290]
    im = Image.fromarray(rgb_crop)
    w, h = im.size
    out_height = HEIGHT 
    im1 = im.resize((int(w * out_height/h), out_height))
    im1.save(str(output_dir / 'completed_scene.png'))

def visualize_results(output_dir, voxel_size, real_world,
                    all_pos, kit_pos,
                    all_p0_vol_rotate, p1_vol, all_sample_sc, all_sample_raw, 
                    sc_preds, raw_preds, tn_infos):
    """
        Passing gui=True will show the pybullet instead of dumping the gifs
    """
    
    obj_num = len(all_pos)
    half_kit_shape = (np.array(p1_vol.shape) / 2).astype(int)
    vol_dir = output_dir / 'vols'
    p1_vol_sc, p1_vol_raw = all_sample_sc[0]['p1_vol'], all_sample_raw[0]['p1_vol']
    p1_mesh_path_gt = dump_vol(vol_dir, 'gt', 'kit_vol', voxel_size, p1_vol)
    p1_mesh_path_sc = dump_vol(vol_dir, 'sc', 'kit_vol', voxel_size, p1_vol_sc)
    p1_mesh_path_raw = dump_vol(vol_dir, 'raw', 'kit_vol', voxel_size, p1_vol_raw)
    gt_labels, sc_labels, raw_labels, tn_labels, user_labels = [], [], [], [], []
    obj_mesh_paths_sc = []
    for i in range(obj_num):
        p0_vol = all_p0_vol_rotate[i]
        p0_vol_sc, p0_vol_raw = all_sample_sc[i]['p0_vol'], all_sample_raw[i]['p0_vol']
        p0_mesh_path_gt =  dump_vol(vol_dir, 'gt', f'obj_vol_{i}', voxel_size, p0_vol)
        p0_mesh_path_sc =  dump_vol(vol_dir, 'sc', f'obj_vol_{i}', voxel_size, p0_vol_sc)
        p0_mesh_path_raw = dump_vol(vol_dir, 'raw', f'obj_vol_{i}', voxel_size, p0_vol_raw)
        obj_mesh_paths_sc.append(p0_mesh_path_sc)
        p1_coords, p1_coords_user, ori_gt = \
            all_sample_sc[i]['p1_coords'], all_sample_sc[i]['p1_coords_user'], all_sample_raw[i]['p1_ori']
        coords_pred_sc, ori_pred_sc = sc_preds[i]
        coords_pred_raw, ori_pred_raw = raw_preds[i]

        pos_gt = (p1_coords - half_kit_shape) * voxel_size
        pos_user = (p1_coords_user - half_kit_shape) * voxel_size
        pos_pred_sc = (coords_pred_sc - half_kit_shape) * voxel_size
        pos_pred_raw = (coords_pred_raw - half_kit_shape) * voxel_size
        pos_gt_tn, pos_pred_tn, ori_pred_tn = tn_infos[i]
        pos_pred_tn = pos_pred_tn - pos_gt_tn + pos_gt

        gt_labels.append((p0_mesh_path_gt, pos_gt, ori_gt))
        sc_labels.append((p0_mesh_path_sc, pos_pred_sc, ori_pred_sc))
        raw_labels.append((p0_mesh_path_raw, pos_pred_raw, ori_pred_raw))
        tn_labels.append((p0_mesh_path_raw, pos_pred_tn, ori_pred_tn))
        user_labels.append((p0_mesh_path_sc, pos_user, [0,0,0,1]))

    save_completed_scene(output_dir, real_world, all_pos, obj_mesh_paths_sc, p1_mesh_path_sc, kit_pos)
    save_around_imgs(output_dir, f'gt', p1_mesh_path_gt, gt_labels)
    save_around_imgs(output_dir, f'user', p1_mesh_path_sc, user_labels)
    save_around_imgs(output_dir, f'sc', p1_mesh_path_sc, sc_labels)
    save_around_imgs(output_dir, f'raw', p1_mesh_path_raw, raw_labels)
    save_around_imgs(output_dir, f'tn', p1_mesh_path_raw, tn_labels)
    
def prepare_sample(sample):
    p0_vol, p1_vol, p1_coords, p1_coords_user, p1_ori, concav_ori, sym = sample.values()
    p0_vol_rotate_ten = torch.tensor(p0_vol).unsqueeze(dim=0)
    p1_vol_ten = torch.tensor(p1_vol).unsqueeze(dim=0)
    user_coords_gt_ten = torch.tensor(p1_coords).unsqueeze(dim=0)
    user_coords_ten = torch.tensor(p1_coords_user).unsqueeze(dim=0)
    p1_ori_gt_ten = torch.tensor(p1_ori).unsqueeze(dim=0)
    batch = {
        "p0_vol": p0_vol_rotate_ten,
        "p1_vol": p1_vol_ten, 
        "p1_coords": user_coords_gt_ten,
        "p1_coords_user": user_coords_ten,
        "p1_ori": p1_ori_gt_ten,
        "concav_ori": torch.tensor([concav_ori]),
        "symmetry": torch.tensor([sym]),
    }
    return batch

def get_preds(batch, transporter, rotator):
    with torch.no_grad():
        _, pred_coords, _, pos_diff = transporter.run(batch, training=False, log=False, calc_loss=True)
        batch['p1_coords'] = pred_coords.astype(int)
        _, _, pred_ori, rot_diff = rotator.run(batch, training=False, log=False, calc_loss=True)
        pred_coords = pred_coords[0]
    torch.cuda.empty_cache()
    return pred_coords, pred_ori, pos_diff, rot_diff

def screenshot_pc(output_dir, suffix, angle, cam_h, sw, sh=350):
    pc_path = str(output_dir / f'input_pc.ply')
    pc_img_path =  str(output_dir / f'input_pc_{suffix}.png')
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.io.read_point_cloud(pc_path)
    vis.add_geometry(pcd)
    angle = np.pi/180 * angle
    length = 1.5
    front = np.array([np.cos(angle) * length, np.sin(angle) * length, cam_h])
    vis.get_view_control().set_front(front)
    up_angle = np.pi/180 * (45)
    vis.get_view_control().set_up([0, np.sin(up_angle), np.cos(up_angle)])
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(pc_img_path)
    vis.destroy_window()
    im = Image.open(pc_img_path)
    im1 = im.crop((sw, sh, sw+700, sh+480))
    im1.save(pc_img_path)
    center_crop_img(pc_img_path)

def make_img_transparent(path):
    img = Image.open(str(path)).convert("RGBA")
    datas = img.getdata()
    newData = []
    for item in datas:
        r, g, b, _ = item
        if (r, g, b) == (0,)*3:
            newData.append((255, 255, 255, 0))
        else:
            newData.append((r, g, b, 255))
    img.putdata(newData)
    img.save(path)
    return path

def draw_boundaries(img):
    red = img[:,:,0]
    height, width, _ = img.shape
    mask = np.zeros((height+2, width+2), np.uint8)
    flooded = red.copy()
    flags = 4 | cv2.FLOODFILL_MASK_ONLY
    cv2.floodFill(flooded.astype(np.float32), mask, (8, 8), 1, 2, 2, flags)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > height*width*0.8 or area < 1000:
            continue
        cv2.drawContours(img, [contour], -1, (255, 255, 255), 3)
    return img

def save_pred_masks(output_dir, img, d, all_pred_masks, kit_mask, realworld):
    mask_threshold = 0.9
    mask_painted = np.zeros_like(img)
    kit_color = np.array([C1[2], C1[1], C1[0]])
    mask_painted[kit_mask==1, :] = kit_color * 255
    for i, msk in enumerate(all_pred_masks):
        r, g, b =  COLORS[i][:3]
        color = np.array([b, g, r]) * 255
        mask_painted[msk >= mask_threshold, :] = color
    mask_painted = cv2.resize(mask_painted.astype(float), (853, 480))
    if realworld:
        mask_painted = mask_painted[:, -720:, :]
    else:
        mask_painted = mask_painted[:, 65:785, :]
    out_path_seg = str(output_dir / 'seg.png')
    cv2.imwrite(out_path_seg, mask_painted)
    make_img_transparent(out_path_seg)
    out_path_seg_bn = str(output_dir / 'seg_bn.png')
    dist_bn = draw_boundaries(mask_painted)
    cv2.imwrite(out_path_seg_bn, dist_bn)
    make_img_transparent(out_path_seg_bn)

    background = Image.open(str(output_dir / 'input_rgb.png'))
    foreground = Image.open(out_path_seg_bn)
    background.paste(foreground, (0, 0), foreground)
    background.save(str(output_dir /'seg_overlay.png'))

def plot_input(output_dir, realworld, img, d, all_pred_masks, p1_mask):
    np.save(output_dir / 'input_rgb.npy', img)
    np.save(output_dir / 'input_d.npy', d)
    save_fig(output_dir, 'input_rgb', img, center_crop=False) # rgb
    rgb_img = np.array(Image.open(str(output_dir / 'input_rgb.png')))
    if realworld:
        rgb_img = rgb_img[:, -720:, :]
    else:
        rgb_img = rgb_img[:, 65:785, :]
    Image.fromarray(rgb_img).save(str(output_dir / 'input_rgb.png'))
    save_fig(output_dir, 'input_d', d, cmap = 'jet', center_crop=False) # d

    save_pred_masks(output_dir, img, d, all_pred_masks, p1_mask, realworld)
    intrinsics = get_intrinsics(realworld)
    sw = 750 # if realworld else 550
    output_filename = output_dir / 'input_pc.ply'
    camera_points, color_points = get_pointcloud_color(img, d, intrinsics)
    color_points = color_points.astype(np.uint8)
    camera_points_f = camera_points[camera_points[:,2]<1.5,:]
    color_points_f = color_points[camera_points[:,2]<1.5,:]
    matrix = R.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()
    camera_points_f = (matrix @ camera_points_f.T).T
    write_pointcloud(output_filename, camera_points_f, color_points_f)
    screenshot_pc(output_dir, '1', -180, 5, sw)
    screenshot_pc(output_dir, '2', -200, 3, sw)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    seed_all_int(cfg.seeds.test)
    voxel_size = cfg.env.voxel_size
    vm_cfg = cfg.vol_match_6DoF 

    TEST = False
    REAL_WORLD = False
    USER_INPUT = not vm_cfg.no_user_input

    dataset_path =  Path('dataset/eval_realworld/snap')  if REAL_WORLD \
                    else Path('dataset/vol_match_abc_all/val') 

    real_suffix = '_realworld' if REAL_WORLD else ''
    log_suffix  = '_test' if TEST else ''
    user_suffix = '_nouser' if not USER_INPUT else ''
    logs_dir = Path(f"res_figs{real_suffix}{log_suffix}{user_suffix}/")
    dataset = ResultDataset.from_cfg(cfg, dataset_path, REAL_WORLD)

    if USER_INPUT:
        sc_transport_path = 'checkpoints/sc_transporter.pth'
        sc_rotate_path = 'checkpoints/sc_rotator.pth'
        raw_transport_path = 'checkpoints/raw_transporter.pth'
        raw_rotate_path = 'checkpoints/raw_rotator.pth'
    else:
        sc_transport_path = 'checkpoints/sc_transporter_full.pth'
        sc_rotate_path = 'checkpoints/sc_rotator_full_alt.pth'
        raw_transport_path = 'checkpoints/raw_transporter_full.pth'
        raw_rotate_path = 'checkpoints/raw_rotator_full.pth'

    sc_transporter = VolMatchTransport.from_cfg(vm_cfg, voxel_size, load_model=True, log=False, model_path=sc_transport_path)
    sc_rotator = VolMatchRotate.from_cfg(vm_cfg, load_model=True, log=False, model_path=sc_rotate_path)
    raw_transporter = VolMatchTransport.from_cfg(vm_cfg, voxel_size, load_model=True, log=False, model_path=raw_transport_path)
    raw_rotator = VolMatchRotate.from_cfg(vm_cfg, load_model=True, log=False, model_path=raw_rotate_path)
    tn_bounds = get_tn_bounds() if REAL_WORLD else None
    transporternet = Transportnet.from_cfg(cfg.evaluate, load_model=True, view_bounds_info=tn_bounds)

    def evaluate_sample(cnt, dataset_int):
        output_dir = logs_dir / f'{cnt}'
        output_dir.mkdir(exist_ok=True, parents=True)
        rgb, d, all_pos, all_pred_masks, p1_mask, kit_pos, p1_vol, \
            all_p0_vol_rotate, all_sample_sc, all_sample_raw, all_sample_tn = dataset[dataset_i]
        obj_num = len(all_pos)
        # num_conditon = obj_num == 2 if USER_INPUT else obj_num != 1
        # num_conditon |= TEST
        # if not num_conditon:
        #     return False

        print(f'Count {cnt}, Evaluate sample {dataset_int}.')
        diffs = []
        sc_preds, raw_preds = [], []
        tn_infos  = []
        for i in range(obj_num):
            sample_sc_ten = prepare_sample(all_sample_sc[i])
            sc_pred_coords, sc_pred_ori, sc_pos_diff, sc_rot_diff = get_preds(sample_sc_ten, sc_transporter, sc_rotator)
            rej_condition = sc_pos_diff > 20
            if USER_INPUT:
                rej_condition = sc_pos_diff > 7 or sc_rot_diff > 7
            if rej_condition:
                print('\t', i, sc_pos_diff, sc_rot_diff)
                return False
            sample_raw_ten = prepare_sample(all_sample_raw[i])
            raw_pred_coords, raw_pred_ori, raw_pos_diff, raw_rot_diff = get_preds(sample_raw_ten, raw_transporter, raw_rotator)
            print('\t', i, sc_pos_diff, sc_rot_diff, raw_pos_diff, raw_pos_diff)
            sample_tn = all_sample_tn[i]
            with torch.no_grad():
                _, (tn_pred_pos, tn_pred_quat), (tn_pos_diff, tn_rot_diff) = transporternet.run(sample_tn, training=False)
                torch.cuda.empty_cache()
            tn_info = [sample_tn[5], tn_pred_pos, tn_pred_quat]
            sc_preds.append([sc_pred_coords, sc_pred_ori])
            raw_preds.append([raw_pred_coords, raw_pred_ori])
            diffs.append([sc_pos_diff, sc_rot_diff, raw_pos_diff, raw_rot_diff, tn_pos_diff, tn_rot_diff])
            tn_infos.append(tn_info)
        
        diffs = np.array(diffs)
        mean_diffs, min_diffs, max_diffs = np.mean(diffs, axis=0), np.min(diffs, axis=0), np.max(diffs, axis=0)
        if USER_INPUT:
            condition = (max_diffs[2] > 7 or max_diffs[3] > 7)
        else:
            condition = True
            print('\t', cnt, obj_num, mean_diffs[:2], max_diffs[-2:])
        if TEST:
            condition = True
        if condition:
            visualize_results(
                output_dir, voxel_size, REAL_WORLD,
                all_pos, kit_pos,
                all_p0_vol_rotate, p1_vol, all_sample_sc, all_sample_raw, 
                sc_preds, raw_preds, tn_infos)
            plot_input(output_dir, REAL_WORLD, rgb, d, all_pred_masks, p1_mask)
            return True
        return False

    cnt, total = 0, 30
    if TEST:
        total = 1
    dataset_ind = 0
    sample_ids = [202,220,225,227,238,243,252,259,261]
    while cnt < total and dataset_ind < len(sample_ids):
        if not USER_INPUT:
            dataset_i = sample_ids[dataset_ind]
            dataset_ind += 1
        else:
            dataset_i = np.random.randint(0, len(dataset))
        if evaluate_sample(cnt, dataset_i):
            cnt += 1
            print(f'Got {cnt}/{total} samples.')

if __name__ == "__main__":
    main()
