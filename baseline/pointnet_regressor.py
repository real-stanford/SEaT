from utils import get_device
import torch
from torch import nn, optim
from omegaconf import DictConfig
import numpy as np
from os.path import join
import random
from learning.pointnet_plus import PointNetPlusEmbed
from learning.quat_rgs_loss import QuatRgsLoss
from utils.rotation import get_quat_diff
from torch.utils.tensorboard import SummaryWriter

class PointnetRegreessor(nn.Module):
    """Rotate module."""

    def __init__(self, part_shape, kit_shape, np_part, np_kit, 
                logs_dir, logger, lite=False):
        """Transport module for placing.

        Args:
            n_rotations: number of rotations of convolving kernel.
            crop_size: crop size around pick argmax used as convolving kernel.
            preprocess: function to preprocess input images.
        """
        super().__init__()

        self.device = get_device()
        self.part_shape = np.array(part_shape)
        self.kit_shape = np.array(kit_shape)
        self.np_part = np_part
        self.np_kit = np_kit

        # paddings for rotations/crops
        self.kit_pad_size = np.array([16,16,16])
        self.kit_padding = ((0,0),) + ((16, 16),) * 3
        
        # logging
        self.logger = logger
        self.logs_dir = logs_dir

        # model
        self.pc_embed = PointNetPlusEmbed(c_in=3, lite=lite).to(self.device).float()
        self.regressor = nn.Sequential(
                            nn.Linear(2048, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
                            nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512),
                            nn.Linear(512, 4),
                        ).to(self.device).float()
        self.crit = QuatRgsLoss()
        params = list(self.pc_embed.parameters()) + list(self.regressor.parameters())
        self.optim = optim.Adam(params, lr=1e-4)
        self.train_step = 0

    @staticmethod
    def from_cfg(vm_cfg: DictConfig, load_model: bool = False):
        p0_vol_shape = vm_cfg.p0_vol_shape
        p1_vol_shape = vm_cfg.p1_vol_shape
        np_part = int(vm_cfg.np_part)
        np_kit = int(vm_cfg.np_kit)
        lite = vm_cfg.lite
        logs_dir = f'baseline/logs/pointnet_regressor'
        logger = SummaryWriter(join(logs_dir, "tensorboard"))
        model = PointnetRegreessor(p0_vol_shape, p1_vol_shape, np_part, np_kit,
                                   logs_dir, logger, lite)
        if load_model:
            model.load(f'{logs_dir}/10000_pointnet_regressor.pth')
        return model

    @staticmethod
    def normalize_pc(pc0, pc1):
        np0 = pc0.shape[0]
        pts = np.concatenate((pc0,pc1), axis=0)
        mean = np.mean(pts, axis=0)
        pts = pts - mean
        m = np.max(np.sqrt(np.sum(pts ** 2, axis=1)))
        pts = pts/m
        pc0 = pts[:np0, :]
        pc1 = pts[np0:, :]
        return pc0.T, pc1.T

    @staticmethod
    def sample_pointcloud_from_tsdf(tsdf, n_pt):
        size = np.array(tsdf.shape) // 2
        x, y, z = np.where(tsdf<=0)
        n_tot = len(x)
        pts = np.array(list(zip(x,y,z)))
        if n_tot == 0: # no point in the region, fill in (-1,-1,-1)?
            sampled_pts = np.zeros((n_pt, 3))-1
        elif len(pts) < n_pt: # no enough pt, , fill in (-1,-1,-1)?
            sampled_pts = np.concatenate([pts, np.zeros((n_pt-n_tot, 3))-1], axis=0)
        else:
            sampled_inds = random.sample(range(pts.shape[0]), n_pt)
            sampled_pts = pts[sampled_inds, :]
        sampled_pts -= size
        return sampled_pts

    def forward(self, pc0, pc1, gt_ori):
        pc0 = torch.tensor(pc0).to(self.device).float()
        pc1 = torch.tensor(pc1).to(self.device).float()
        gt_ori = gt_ori.to(self.device).float()
        pc0_embed = self.pc_embed(pc0)
        pc1_embed = self.pc_embed(pc1)
        embeds = torch.cat([pc0_embed, pc1_embed], dim=1)
        
        pred_ori = self.regressor(embeds)
        loss = self.crit(pred_ori, gt_ori, pc0) 
        return loss, pred_ori

    def run(self, sample, training=False, log=True):
        self.train(training)
        # prepare data
        p0_vol, p1_vol, p1_coords, p1_ori = sample.values()
        p1_vol_pad = np.pad(p1_vol, self.kit_padding, constant_values=1)
        nb = p0_vol.shape[0]
        pc0s, pc1s = np.zeros((nb, 3, self.np_part)), np.zeros((nb, 3, self.np_kit))
        # crop kit volume and sample point clouds
        for i in range(nb):
            p_gt = p1_coords[i]
            x, y, z = p_gt + self.kit_pad_size
            t, h, w = self.kit_shape // 2
            p1_vol_crop = p1_vol_pad[i, x-t:x+t, y-h:y+h, z-w:z+w]
            pc0 = self.sample_pointcloud_from_tsdf(p0_vol[i], self.np_part)
            pc1 = self.sample_pointcloud_from_tsdf(p1_vol_crop, self.np_kit)
            pc0_norm, pc1_norm = self.normalize_pc(pc0, pc1)
            pc0s[i] = pc0_norm
            pc1s[i] = pc1_norm
        # get prediction and loss
        loss, quat_pred = self.forward(pc0s, pc1s, p1_ori)
        ori_diff = np.mean(get_quat_diff(p1_ori.cpu().detach().numpy(), 
                                        quat_pred.cpu().detach().numpy()) * 180/np.pi)
        # logging
        if log and training:
            print_str = f'step {self.train_step+1:04d}   '
            print_str += f'loss {loss:.3f} '
            print_str += f'ori_diff: {ori_diff:.3f} '
            print(print_str)
            self.log(loss, ori_diff)
        # update params
        if training:
            self.backward(loss)
        return loss, p_gt, quat_pred, ori_diff
                
    def log(self, total_loss, ori_diff):
        if self.logger == None:
            return
        log_dic = {
            'total_loss': total_loss,
            'ori_diff': ori_diff,
        }
        for k, v in log_dic.items():
            self.logger.add_scalar(k, v, self.train_step)

    def backward(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.train_step += 1
    
    def load(self, model_fname):
        checkpoint = torch.load(model_fname, map_location=self.device)
        self.pc_embed.load_state_dict(checkpoint['pc_embed'])
        self.regressor.load_state_dict(checkpoint['regressor'])
        self.train_step = checkpoint['train_step']

    def save(self):
        state = {'pc_embed': self.pc_embed.state_dict(),
                'regressor': self.regressor.state_dict(),
                'train_step': self.train_step}
        path = f"{self.logs_dir}/{self.train_step}_pointnet_regressor.pth"
        torch.save(state, path)
        return path