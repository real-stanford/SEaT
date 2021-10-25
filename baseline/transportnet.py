from baseline.resnet import ResNet43_8s
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms, models
import PIL

from utils import get_device, get_pos_from_pix, get_pix_size, get_pix_from_pos
from utils.rotation import euler_to_quat, get_quat_diff_sym, quat_to_euler, quat_to_normal, multiply_quat
from os.path import join
from torch.utils.tensorboard import SummaryWriter
import random

def regressor():
    return nn.Sequential(
                nn.Linear(1, 32), nn.ReLU(),
                nn.Linear(32, 32), nn.ReLU(),
                nn.Linear(32, 1))

class Transport(nn.Module):
    """Matching module."""
    def __init__(self, num_rotations, crop_size, vicinity):
        super(Transport, self).__init__()

        self.device = get_device()

        self.n_rotations = num_rotations
        self.kernel_dim = 27
        # self.kernel_dim = 3

        self.querynet = ResNet43_8s(6, self.kernel_dim).to(self.device).float()
        self.keynet = ResNet43_8s(6, self.kernel_dim).to(self.device).float()
        params = list(self.querynet.parameters())+list(self.keynet.parameters())

        self.z_regressor = regressor().to(self.device).float()
        self.roll_regressor = regressor().to(self.device).float()
        self.pitch_regressor = regressor().to(self.device).float()
        params += list(self.z_regressor.parameters()) + \
            list(self.roll_regressor.parameters())+list(self.pitch_regressor.parameters())

        self.optim = optim.Adam(params, lr=1e-4)
        self.ce_loss= nn.CrossEntropyLoss()
        self.huber_loss = nn.SmoothL1Loss()
        
        self.crop_size = crop_size
        self.vicinity = vicinity
        self.padding = (self.crop_size//2,)*4
        self.pad_size = [self.crop_size//2, self.crop_size//2]

    def forward(self, img_obj, img_kit, p, concav_ori, syms, q_user, quat_user):
        img_obj = nn.functional.pad(img_obj, self.padding).to(self.device).float()
        img_kit = nn.functional.pad(img_kit, self.padding).to(self.device).float()
        logits = self.keynet(img_kit) # (1, self.kernel_dim, H, W)
        kernel_raw = self.querynet(img_obj) # (1, self.kernel_dim, H, W)
        hs, he, ws, we = p[0], p[0]+self.crop_size, p[1], p[1]+self.crop_size
        kernel_crop = kernel_raw[:, :, hs:he, ws:we] # (1, self.kernel_dim, crop_size, crop_size)
        angles = [i*360/self.n_rotations for i in range(self.n_rotations)]
        kernels = ()
        for angle in angles:
            rotated_kernel = transforms.functional.affine(kernel_crop, angle=angle, translate=[0,0], scale=1, shear=[0,0])
            kernels += (rotated_kernel,)
        kernels = torch.cat(kernels, dim=0) # (n_rot, self.kernel_dim, crop_size, crop_size)
        kernel_paddings = (0,1,0,1)
        kernels = nn.functional.pad(kernels, kernel_paddings, mode='constant')

        output = nn.functional.conv2d(logits[:,:3,:,:], kernels[:,:3,:,:]) # (1, n_rot, H, W)
        output = output.permute(0,2,3,1).squeeze(0) # (H, W, n_rot)
        z_tensor = nn.functional.conv2d(logits[:,3:11,:,:], kernels[:,3:11,:,:]) # (1, n_rot, H, W)
        z_tensor = z_tensor.permute(0,2,3,1).squeeze(0) # (H, W, n_rot)
        roll_tensor = nn.functional.conv2d(logits[:,11:19,:,:], kernels[:,11:19,:,:]) # (1, n_rot, H, W)
        roll_tensor = roll_tensor.permute(0,2,3,1).squeeze(0) # (H, W, n_rot)
        pitch_tensor = nn.functional.conv2d(logits[:,19:27,:,:], kernels[:,19:27,:,:]) # (1, n_rot, H, W)
        pitch_tensor = pitch_tensor.permute(0,2,3,1).squeeze(0) # (H, W, n_rot)

        if self.vicinity != -1:
            # vicinity searching: position
            sH, sW = np.maximum(q_user - self.vicinity, np.zeros(2)).astype(int)
            eH, eW = np.minimum(q_user + self.vicinity, np.array(output.shape[:2])-1).astype(int)
            mask = np.zeros(output.shape)
            mask[sH:eH,sW:eW, :] = 1
            output[mask==0] = -np.inf
            # vicinity searching: orientation
            angles = [ np.pi*2/self.n_rotations * i for i in range(self.n_rotations)]
            quats = [euler_to_quat([0,0,theta], degrees=False) for theta in angles]
            diffs = [get_quat_diff_sym(quat_user, quat, concav_ori, syms) for quat in quats]
            mask = np.zeros(output.shape)
            mask[:,:,np.argmin(diffs)] = 1
            output[mask==0] = -np.inf

        return output, z_tensor, roll_tensor, pitch_tensor

    def run(self, img_obj, img_kit, 
            p, q, theta, z, roll, pitch, 
            concav_ori, syms,
            q_user, quat_user, train=True):
        self.train(train)
        logits, z_tensor, roll_tensor, pitch_tensor = \
            self.forward(img_obj, img_kit, p, concav_ori, syms, q_user, quat_user)
        # Get label.
        # itheta = theta / (360 / self.n_rotations)
        # itheta = np.int64(np.round(itheta)) % self.n_rotations
        # label = logits.shape[2] *(logits.shape[1] * q[0] + q[1]) + itheta
        # label = torch.tensor(label, dtype=torch.int64).reshape(1,)
        # label = label.to(self.device)

        # Get loss and pred.
        # cls_loss = self.ce_loss(logits.reshape(1, -1), label)
        logits_cpu = logits.detach().cpu().numpy()
        pred = np.unravel_index(np.argmax(logits_cpu), logits.shape)
        pred_q, pred_itheta = pred[:2], pred[2]
        pred_theta = pred[2] / self.n_rotations * 360

        # Use a window for regression rather than only exact.
        # H, W, n_rot = z_tensor.shape
        # u_window = 16
        # v_window = 16
        # theta_window = 1
        # u_min = max(q[0] - u_window, 0)
        # u_max = min(q[0] + u_window + 1, H)
        # v_min = max(q[1] - v_window, 0)
        # v_max = min(q[1] + v_window + 1, W)
        # theta_min = max(itheta - theta_window, 0)
        # theta_max = min(itheta + theta_window + 1, n_rot)

        def regress(inp, regressor, gt):
            los=None
            # est = inp[u_min:u_max, v_min:v_max, theta_min:theta_max]
            # est = est.reshape(-1, 1)
            # est = regressor(est)
            # gt = [gt] * est.shape[0]
            # lab = torch.tensor(gt).reshape(-1,1).to(self.device).float()
            # los = self.huber_loss(est, lab)
            pred = regressor(inp[pred_q[0], pred_q[1], pred_itheta].reshape(1,1))
            pred = pred[0].cpu().detach().item()
            return los, pred

        z_loss, pred_z = regress(z_tensor, self.z_regressor, z)
        roll_loss, pred_roll = regress(roll_tensor, self.roll_regressor, roll)
        pitch_loss, pred_pitch = regress(pitch_tensor, self.pitch_regressor, pitch)
        
        # loss = cls_loss + 10 * (z_loss + roll_loss + pitch_loss)

        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        return 0, (pred_q, pred_theta, pred_z, pred_roll, pred_pitch)

class Transportnet(nn.Module):
    def __init__(self, view_bounds_obj, view_bounds_kit, pix_size, vicinity):
        super(Transportnet, self).__init__()

        self.num_rotations = 36
        self.crop_size = 96
        self.vicinity = vicinity

        self.device = get_device()
        user_info_str = "_user" if self.vicinity != -1 else ""
        self.logs_dir = f"baseline/logs/transportnet{user_info_str}"
        self.logger = SummaryWriter(join(self.logs_dir, "tensorboard"))
        self.train_step = 0

        self.view_bounds_obj = view_bounds_obj
        self.view_bounds_kit = view_bounds_kit
        self.pix_size = pix_size

        self.transport_model = Transport(self.num_rotations, self.crop_size, self.vicinity)
    
    @staticmethod
    def from_cfg(cfg, load_model=False, view_bounds_info = None):
        image_size = np.array(cfg.image_size)
        if view_bounds_info is None:
            view_bounds = np.array(cfg.workspace_bounds)
            view_bounds_obj = np.array(cfg.workspace_bounds_obj)
            view_bounds_kit = np.array(cfg.workspace_bounds_kit)
        else:
            view_bounds, view_bounds_obj, view_bounds_kit = view_bounds_info
        vicinity = int(cfg.vicinity)
        pix_size = get_pix_size(view_bounds, image_size[0])
        model = Transportnet(view_bounds_obj, view_bounds_kit, pix_size, vicinity)
        if load_model:
            model.load(cfg.model_path)
        return model

    def preprocess(self, cmap, hmap):
        cmap, hmap = torch.tensor(cmap), torch.tensor(hmap)
        cmap = cmap.permute(2,0,1).unsqueeze(0)
        hmap = hmap.unsqueeze(0).unsqueeze(0)
        hmap = hmap.repeat(1,3,1,1)
        image = torch.cat([cmap, hmap], dim=1)
        color_mean = 60
        depth_mean = 0.002
        color_std = 60
        depth_std = 0.008
        image[:, :3, :, :] = (image[:, :3, :, :] / 255 - color_mean) / color_std
        image[:, 3:, :, :] = (image[:, 3:, :, :] - depth_mean) / depth_std
        return image

    def run(self, sample, training=False, log=True):
        self.train(training)
        cmap_obj, hmap_obj, cmap_kit, hmap_kit, \
            pick_pos, place_pos, place_ori, \
            concav_ori, syms, user_pos, user_quat = sample 
        
        p0_pix = get_pix_from_pos(pick_pos, self.view_bounds_obj, self.pix_size)
        p1_pix = get_pix_from_pos(place_pos, self.view_bounds_kit, self.pix_size)
        p1_pix_user = get_pix_from_pos(user_pos, self.view_bounds_kit, self.pix_size)
        import matplotlib.pyplot as plt
        cmap_obj[p0_pix[0]-10:p0_pix[0]+10, p0_pix[1]-10:p0_pix[1]+10, :] = 200
        z = place_pos[2]
        
        roll, pitch, yaw = quat_to_euler(place_ori, degrees=True) # degrees
        img_obj = self.preprocess(cmap_obj, hmap_obj)
        img_obj = img_obj.to(self.device).float() # (1, 6, h, w)
        img_kit = self.preprocess(cmap_kit, hmap_kit)
        img_kit = img_kit.to(self.device).float() # (1, 6, h, w)

        # Compute training loss.
        loss, preds = self.transport_model.run(
            img_obj, img_kit, 
            p0_pix, p1_pix, yaw, z, roll, pitch, 
            concav_ori, syms,
            p1_pix_user, user_quat, training)
        pred_q, pred_theta, pred_z, pred_roll, pred_pitch = preds
        pred_pos = get_pos_from_pix(pred_q, self.view_bounds_kit, self.pix_size)
        pred_pos[2] = pred_z
        pred_quat = euler_to_quat([pred_roll, pred_pitch, pred_theta], degrees=True)
        pos_diff = np.linalg.norm(place_pos-pred_pos) * 1000 # mm
        ori_diff = get_quat_diff_sym(place_ori, pred_quat, concav_ori, syms) # deg

        if training:
            self.train_step += 1

        if training and log:
            self.logger.add_scalar('loss', loss, self.train_step)
            self.logger.add_scalar('pos_diff', pos_diff, self.train_step)
            self.logger.add_scalar('quat_diff', ori_diff, self.train_step)
        
        return loss, (pred_pos, pred_quat), (pos_diff, ori_diff)
        
    def load(self, path):
        print(f'Loaded from {path}')
        state = torch.load(path)
        self.transport_model.load_state_dict(state['transport_model'])

    def save(self):
        state = {'transport_model': self.transport_model.state_dict()}
        path = f'{self.logs_dir}/model_{self.train_step}.pth'
        torch.save(state, path)
        print(f"Model saved at path: {path}")

if __name__ == "__main__":
    
    cmap = torch.rand(1,128,144,3)
    hmap = torch.rand(1,128,144)
    p = [[5,5]]
    q = [[1,1]]
    theta = np.random.random()
    sample = (cmap, hmap, p, q, [5], [0.02], [10], [15])

    view_bounds = np.array([ [0, 0.7], [-0.7, 0.7], [0.00, 0.5] ])
    pix_size = 0.7/640
    model = Transportnet(view_bounds, pix_size)

    for i in range(1):
        loss, pred = model.run(sample, training=True, log=True)
        print(loss)
        print(pred)

    model.save()
    model.load()