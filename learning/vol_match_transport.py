# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transport module."""

import numpy as np
from numpy.linalg import norm
import torch
from torch import nn, optim
import torch.nn.functional as F
from omegaconf import DictConfig
from pathlib import Path
from utils import get_device, init_logs_dir, center_crop
from utils.torch_utils import weight_init
from torch.utils.tensorboard import SummaryWriter

class VolEncoder(nn.Module):
    """
        in_shape: (B, C, T, H, W)    
        out_shape: (B, C, T, H, W) 
    """
    def __init__(self, kernel_dim, init_weight=False, lite=False):
        """Initialize the layers of the network as instance variables."""
        super(VolEncoder, self).__init__()
        self.lite=lite
        self.maxpool = nn.MaxPool3d(2, 2)

        self.convd1 = nn.Conv3d(1, 16, 3, stride=1, padding=1)
        self.convd2 = nn.Conv3d(16, 32, 3, stride=2, padding=1)
        self.convd3 = nn.Conv3d(32, 64, 3, stride=2, padding=1)
        self.convd4 = nn.Conv3d(64, 128, 3, stride=2, padding=1)

        self.convb_lite = nn.Conv3d(64, 128, 3, stride=2, padding=1)
        self.convb = nn.Conv3d(128, 256, 3, stride=2, padding=1)

        self.convu1 = nn.Conv3d(384, 128, 3, stride=1, padding=1)
        self.convu2 = nn.Conv3d(192, 64, 3, stride=1, padding=1)
        self.convu3 = nn.Conv3d(96, 32, 3, stride=1, padding=1)
        self.convf = nn.Conv3d(32, kernel_dim, 1, stride=1, padding=0)

        if init_weight:
            self.apply(weight_init)

    def forward(self, x):
        
        xd1 = F.relu(self.convd1(x))
        xd2 = F.relu(self.convd2(xd1))
        xd3 = F.relu(self.convd3(xd2))
        if not self.lite:
            xd4 = F.relu(self.convd4(xd3))
            xb = F.relu(self.convb(xd4))
            xu1 = F.relu(self.convu1(torch.cat((xd4, F.interpolate(xb, scale_factor=(2,2,2))), 1)))
        else:
            xu1 = F.relu(self.convb_lite(xd3))

        xu2 = F.relu(self.convu2(torch.cat((xd3, F.interpolate(xu1, scale_factor=(2,2,2))), 1)))
        xu3 = F.relu(self.convu3(torch.cat((xd2, F.interpolate(xu2, scale_factor=(2,2,2))), 1)))

        output = self.convf(xu3)
        return output

class VolMatchTransport(nn.Module):
    """Transport module."""

    def __init__(self, 
                max_perturb_delta, part_shape, kit_shape, vox_size, kit_padding,
                logs_dir, logger, 
                dist_aware_loss, lite=False, upsample=False, device=None):
        """Transport module for placing.

        Args:
            max_perturb_delta: max purturbations (number of voxels)
        """
        super().__init__()
        # print(f'transporter upsample : {upsample}')
        self.device = get_device() if device is None else device

        self.part_shape = np.array(part_shape)
        self.kit_shape = np.array(kit_shape)
        self.vox_size = vox_size * 1000 # unit is mm
        self.kit_padding = kit_padding
        # make sure shapes are multiple of 16, and kit is larger than obj
        assert np.all(self.part_shape%16==0) and np.all(self.kit_shape%16==0), \
            f'expected kit and obj shapes to be multiples of 4 but get {part_shape}, {kit_shape}'
        assert np.all(self.part_shape < self.kit_shape), \
            f'expected obj shape to be smaller than kit shape but get {part_shape} >= {kit_shape}'

        # max perturbations
        self.max_perturb_delta = np.array(max_perturb_delta)
        # kernel dimention for cross-conv
        self.kernel_dim = 6
        # setup
        self.lite = lite
        self.upsample = upsample

        # volume encoders and optimizer
        self.querynet = VolEncoder(self.kernel_dim, lite=lite).to(self.device)
        self.keynet = VolEncoder(self.kernel_dim, lite=lite).to(self.device)
        params = list(self.querynet.parameters()) + list(self.keynet.parameters())
        if self.upsample:
            self.conv_upsample = nn.Conv3d(1, 16, 3, stride=1, padding=1).to(self.device)
            self.conv_out = nn.Conv3d(16, 1, 1, stride=1, padding=0).to(self.device)
            params += list(self.conv_upsample.parameters()) + list(self.conv_out.parameters())
        self.optim = optim.Adam(params, lr=1e-4)
        self.dist_aware_loss = dist_aware_loss

        self.logs_dir = logs_dir
        self.logger = logger
        self.train_step = 0

    @staticmethod
    def from_cfg(vm_cfg: DictConfig, vox_size: float, load_model: bool = False, log=True, model_path=None):
        p0_vol_shape = vm_cfg.p0_vol_shape_transport
        p1_vol_shape = vm_cfg.p1_vol_shape_transport
        max_perturb_delta = vm_cfg.max_perturb_delta
        no_user_input = vm_cfg.no_user_input
        kit_padding = (0,)*6
        if no_user_input:
            kit_padding = (8,)*6
            p1_vol_shape = [400,400,256]
            max_perturb_delta = np.array([144, 144, 88])
        dist_aware_loss = vm_cfg.dist_aware_loss
        lite = vm_cfg.lite
        upsample = vm_cfg.upsample
        upsample_str = "_up" if upsample else ""
        logs_dir, logger = None, None
        if log:
            logs_dir = init_logs_dir(vm_cfg, f'{vm_cfg.vol_type}_transport{upsample_str}_full')
            logger = SummaryWriter(logs_dir / "tensorboard")
        model = VolMatchTransport(max_perturb_delta, p0_vol_shape, p1_vol_shape, vox_size, kit_padding,
                                  logs_dir, logger, dist_aware_loss, lite, upsample)
        if load_model:
            if model_path is None:
                model.load(Path(vm_cfg.transporter_path))
            else:
                model.load(Path(model_path))
        return model

    def translate(self, part_vol, kit_vol, train=False):
        self.train(train)
        batch_size = part_vol.shape[0]
        kernel = self.querynet(part_vol) # (B, self.kernel_dim, t, h, w)
        kit_vol = F.pad(kit_vol, self.kit_padding, "constant", 1)
        logits = self.keynet(kit_vol) # (B, self.kernel_dim, T, H, W)
        kernel_paddings = (0,1,)*3
        kernel = F.pad(kernel, kernel_paddings, mode='constant', value=0)
        half_conv_size = (self.part_shape//2 + self.max_perturb_delta)//2
        sH, sW, sD = np.array(logits.shape[2:5])//2 - half_conv_size
        eH, eW, eD = np.array(logits.shape[2:5])//2 + half_conv_size
        logits = logits[:,:,sH:eH,sW:eW,sD:eD]
        output = F.conv3d(logits.view(1, batch_size*self.kernel_dim, *logits.shape[2:5]), 
                          kernel, groups=batch_size) # (1, B, T-t, H-h, W-w)
        output = output.squeeze(0) # (B, T-t, H-h, W-w)
        if self.upsample:
            output = output.unsqueeze(1) # (B, 1, T-t, H-h, W-w)
            output = F.relu(self.conv_upsample(output))
            output = self.conv_out(F.interpolate(output, scale_factor=(2,2,2)))
            output = output.squeeze(1)
        return output
    
    def forward(self, part_vol, kit_vol, train):
        self.train(train)
        part_vol, kit_vol = part_vol.to(self.device).float(), kit_vol.to(self.device).float()
        translate_logits = self.translate(part_vol, kit_vol, train=train)
        # get predictions
        batch_size, search_shape = translate_logits.shape[0], translate_logits.shape[1:4]
        p_crop_pred = np.zeros((batch_size, 3))
        for i in range(batch_size):
            index = torch.argmax(translate_logits[i]).detach().cpu()
            p_crop_pred[i] = np.unravel_index(index, search_shape)
            # lgts = translate_logits[i].detach().cpu().numpy()
            # p_crop_pred[0] = np.unravel_index(np.argmax(lgts), search_shape)
            # for j in range(1,3):
            #     x, y, z = np.arange(0, search_shape[0]), np.arange(0, search_shape[1]), np.arange(0, search_shape[2])
            #     cx, cy, cz = p_crop_pred[j-1]
            #     r = 50
            #     mask = (x[:,np.newaxis, np.newaxis,]-cx)**2 + (y[np.newaxis, :, np.newaxis]-cy)**2 + (z[np.newaxis, np.newaxis, :]-cz)**2 < r**2
            #     lgts[mask] = 0
            #     p_crop_pred[j] = np.unravel_index(np.argmax(lgts), search_shape)
        scale_factor = 1 if self.upsample else 2
        p_pred = p_crop_pred * scale_factor \
            + self.kit_shape[np.newaxis, Ellipsis]//2 \
            - self.max_perturb_delta[np.newaxis, Ellipsis]
        return translate_logits, p_pred

    def calc_loss(self, p_gt, p_pred, translate_logits):
        batch_size, search_shape = translate_logits.shape[0], translate_logits.shape[1:4]
        # calculate gt position in cropped space
        scale_factor = 1 if self.upsample else 2
        p_crop_gt = (p_gt - self.kit_shape[np.newaxis, Ellipsis]//2 + self.max_perturb_delta) // scale_factor
        p_gt_ind = np.zeros(batch_size)
        for i in range(batch_size):
            x, y, z = p_crop_gt[i]
            p_gt_ind[i] = (x*search_shape[1] + y)*search_shape[2] + z
        p_gt_ind = torch.tensor(p_gt_ind).to(self.device).long()
        loss = F.cross_entropy(translate_logits.reshape(batch_size,-1), p_gt_ind)
        gamma = 1
        if self.dist_aware_loss:
            diff = p_gt - p_pred
            avg_norm = sum([norm(v) for v in diff])/diff.shape[0]
            factor = torch.tensor(avg_norm/norm(self.max_perturb_delta))
            gamma = (torch.exp(factor)-1)
        return loss*gamma
    
    def run(self, sample, training=False, log=True, calc_loss=True):
        p0_vol_full, p1_vol_full, p1_coords, p1_coords_user, p1_ori, _, _ = sample.values()
        if training: # return on gt out of range
            half_search_size = self.max_perturb_delta[np.newaxis, Ellipsis]
            val = p1_coords - p1_coords_user + half_search_size
            if (val<0).any() or ((val-half_search_size*2)>=0).any():
                # print(f'Transport: invalid gt: {val}')
                return None, None, None, None
        # crop kit volumes around user provided centers
        batch_size = p0_vol_full.shape[0]
        p0_vol = torch.empty(batch_size, *self.part_shape)
        p1_vol = torch.empty(batch_size, *self.kit_shape)
        part_crop_center = np.array(p0_vol_full.shape[1:4])//2
        if torch.is_tensor(p1_coords_user):
            p1_coords_user = p1_coords_user.detach().cpu().numpy()
        if torch.is_tensor(p1_coords):
            p1_coords = p1_coords.detach().cpu().numpy()
        for i in range(batch_size):
            p0_vol[i] = center_crop(p0_vol_full[i], part_crop_center, self.part_shape)
            p1_vol[i] = center_crop(p1_vol_full[i], p1_coords_user[i], self.kit_shape)

        p0_vol, p1_vol = p0_vol.unsqueeze(1), p1_vol.unsqueeze(1) # add in channel dim
        translate_logits, p_pred = self.forward(p0_vol, p1_vol, train=training)
        p_gt = None
        if p1_coords is not None:
            p_gt = p1_coords - p1_coords_user + self.kit_shape[np.newaxis, Ellipsis]//2
        p_diff, loss_cpu = None, None
        if p_gt is not None:
            diff = p_gt - p_pred
            p_diff  = sum([norm(v) for v in diff])/diff.shape[0]
        if calc_loss:
            loss = self.calc_loss(p_gt, p_pred, translate_logits)
            loss_cpu = loss.cpu().detach().item()
            if training:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.train_step += 1
            if log:
                # print(f'p_diff: {p_diff:.3f}, p_gt: {p_gt}, p_pred: {p_pred}')
                self.log(loss_cpu, p_diff)
            p_diff *= self.vox_size       

        p_pred = p_pred + p1_coords_user - np.array(self.kit_shape//2)
        return loss_cpu, p_pred, None, p_diff

    def log(self, loss, p_diff):
        if self.logger == None:
            return
        log_dic = {
            'total_loss': loss,
            'p_diff': p_diff,
        }
        for k, v in log_dic.items():
            self.logger.add_scalar(k, v, self.train_step)

    def load(self, transport_fname):
        print(f'Loaded from {transport_fname}')
        checkpoint = torch.load(transport_fname, map_location=self.device)
        self.keynet.load_state_dict(checkpoint['keynet_state_dict'])
        self.querynet.load_state_dict(checkpoint['querynet_state_dict'])
        if self.upsample:
            self.conv_upsample.load_state_dict(checkpoint['conv_upsample'])
            self.conv_out.load_state_dict(checkpoint['conv_out'])
        self.train_step = checkpoint['train_step']

    def save(self):
        state = {'keynet_state_dict': self.keynet.state_dict(),
                'querynet_state_dict': self.querynet.state_dict(),
                'train_step': self.train_step}
        if self.upsample:
            state['conv_upsample'] = self.conv_upsample.state_dict()
            state['conv_out'] = self.conv_out.state_dict()
        path = self.logs_dir/f"{self.train_step}_transporter.pth"
        torch.save(state, path)
        print(f"Model saved at path: {path}")

if __name__ == "__main__":
    model = VolEncoder(3, lite=True)
    volume = torch.rand(1,1,32,32,32)
    out = model(volume)
    print(out.shape)
    model = VolEncoder(3, lite=False)
    volume = torch.rand(1,1,32,32,32)
    out = model(volume)
    print(out.shape)