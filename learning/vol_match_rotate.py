from utils import get_device, distance_quaternion, init_logs_dir, center_crop
from utils.rotation import multiply_quat, invert_quat, quat_to_normal, quat_to_euler, sample_rot, sample_rot_roll, get_quat_diff, get_quat_diff_sym
from utils.torch_utils import batch_run
import torch
from torch import nn, optim
from torch.nn import functional as F
from omegaconf import DictConfig
from pathlib import Path
import numpy as np
from os.path import split, join
import random
from learning.pointnet import PointNetfeat
from learning.pointnet_plus import PointNetPlusEmbed
from learning.quat_rgs_loss import QuatRgsLoss
from torch.utils.tensorboard import SummaryWriter

class ClassifyPC(nn.Module):
    def __init__(self, logs_dir, pointnet_type='pointnet', 
                dist_aware_loss=False, regression=False, temperature=1,
                lite=False):
        super().__init__()
        self.device = get_device()

        self.logs_dir = logs_dir
        self.dist_aware_loss = dist_aware_loss
        self.regression = regression
        self.temperature = temperature
        self.w = 1  

        self.pointnet_type = pointnet_type
        if pointnet_type == 'pointnet':
            self.pc_embed = PointNetfeat(c_in=4, global_feat=True).to(self.device).float()
        elif pointnet_type == 'pointnet2':
            self.pc_embed = PointNetPlusEmbed(c_in=4, lite=lite).to(self.device).float()
        self.classifier = nn.Sequential(
                            nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512),
                            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256),
                            nn.Linear(256, 1),
                        ).to(self.device).float()
        params = list(self.pc_embed.parameters()) + list(self.classifier.parameters())
        if self.regression:
            self.regressor = nn.Sequential(
                                nn.Linear(1024, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
                                nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512),
                                nn.Linear(512, 4),
                            ).to(self.device).float()
            self.rgs_crit = QuatRgsLoss()
            params += list(self.regressor.parameters())
        self.optim = optim.Adam(params, lr=1e-4)
        self.train_step = 0
    
    def get_reg_gt(self, quat_gt, quat_ancs):
        quat_rgs_gt = np.empty_like(quat_ancs)
        for i in range(quat_ancs.shape[0]):
            quat_rgs_gt[i, :] = multiply_quat(quat_gt, invert_quat(quat_ancs[i]))
        return torch.tensor(quat_rgs_gt).to(self.device)
        
    def calc_loss(self, pc_embeds, quat_gt, lab, concav_ori, syms, quats, pc0, calc_loss):
        loss, aux = None, None
        # classification loss
        logits = self.classifier(pc_embeds).reshape(1, -1)
        ind_pred = torch.argmax(logits)
        quat_cls = quats[ind_pred]
        logits /= self.temperature
        quat_pred = quat_cls
        if calc_loss:
            lab = torch.tensor(lab).reshape(1,).to(self.device).long()
            loss_cls = F.cross_entropy(logits, lab)
            if self.dist_aware_loss:
                dists = np.array([get_quat_diff_sym(quat_gt, quat, concav_ori, syms) for quat in quats])
                factor = torch.tensor(dists[ind_pred]/np.max(dists))
                loss_cls *= torch.exp(factor)-1
            loss, quat_pred = loss_cls, quat_cls
        # regression loss: want quat_rgs_pred[ind_pred] * quat_cls = quat_gt
        aux = None
        if self.regression:
            quat_rgs_pred = self.regressor(pc_embeds)
            quat_rgs_gt = self.get_reg_gt(quat_gt, quats)
            quat_pred = multiply_quat(quat_rgs_pred[ind_pred], quat_cls)
            if calc_loss:
                loss_rgs = self.rgs_crit(quat_rgs_pred, quat_rgs_gt, pc0) 
                loss = loss_cls + loss_rgs * self.w
                aux = {'loss_cls': loss_cls, "loss_rgs": loss_rgs*self.w}
        return loss, quat_pred, aux
   
    def forward(self, pc0, pc1, quat_gt, lab, concav_ori, syms, quats, calc_loss):
        # add in additional channel
        pc0 = np.concatenate([pc0, np.zeros((pc0.shape[0], 1, pc0.shape[2]))-1], axis=1)
        pc1 = np.concatenate([pc1, np.ones((1, pc1.shape[1]))], axis=0)
        pc1_repeats = np.repeat(pc1[np.newaxis, Ellipsis], pc0.shape[0], axis=0)
        # concatenate and embed
        pc0_ten = torch.tensor(pc0).to(self.device).float()
        pc1_ten = torch.tensor(pc1_repeats).to(self.device).float()
        pcs = torch.cat([pc0_ten, pc1_ten], dim=2)
        fn_run = lambda pc_batch: self.pc_embed(pc_batch)
        pc_embeds = batch_run(pcs, fn_run, batch_size=20)
        # get loss
        sample_inds = random.sample(range(2048), 512)
        pc = pc0_ten[0,:3,sample_inds]
        loss, quat_pred, aux = self.calc_loss(pc_embeds, quat_gt, lab, concav_ori, syms, quats, pc, calc_loss)
        
        return loss, quat_pred, aux
    
    def backward(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.train_step += 1
    
    def load(self, model_fname):
        checkpoint = torch.load(model_fname, map_location=self.device)
        self.pc_embed.load_state_dict(checkpoint['pc_embed'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        if self.regression:
            self.regressor.load_state_dict(checkpoint['regressor'])
        self.train_step = checkpoint['train_step']

    def save(self):
        state = {'pc_embed': self.pc_embed.state_dict(),
                'classifier': self.classifier.state_dict(),
                'train_step': self.train_step}
        if self.regression:
            state['regressor'] = self.regressor.state_dict()
        path = self.logs_dir/f"{self.train_step}_rotator.pth"
        torch.save(state, path)
        return path

class MatchPC(nn.Module):
    def __init__(self, logs_dir, pointnet_type='pointnet', 
                dist_aware_loss=False, margin=1,
                share_weights=True, proj_head=False):
        super().__init__()
        self.device = get_device()

        self.logs_dir = Path(logs_dir)
        self.pointnet_type = pointnet_type
        self.dist_aware_loss = dist_aware_loss
        self.margin = margin
        self.share_weights = share_weights
        self.proj_head = proj_head
        
        if self.pointnet_type == 'pointnet':
            self.pc_embed0 = PointNetfeat(c_in=3, global_feat=True).to(self.device).float()
            self.pc_embed1 = self.pc_embed0
            if not self.share_weights:
                self.pc_embed1 = PointNetfeat(c_in=3, global_feat=True).to(self.device).float()
        elif self.pointnet_type == 'pointnet2':
            self.pc_embed0 = PointNetPlusEmbed(c_in=3).to(self.device).float()
            self.pc_embed1 = self.pc_embed0
            if not self.share_weights:
                self.pc_embed1 = PointNetPlusEmbed(c_in=3).to(self.device).float()
        if self.proj_head:
            self.proj_head = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
            ).to(self.device).float()

        params = list(self.pc_embed0.parameters())
        if not share_weights:
            params += list(self.pc_embed1.parameters())
        if proj_head:
            params += list(self.proj_head.parameters())
        self.optim = optim.Adam(params, lr=1e-4)
        self.crit = nn.TripletMarginLoss(margin=self.margin, p=2)
        self.train_step = 0
    
    def calc_loss(self, pc0_embeds, pc1_embed, lab):
        lab = torch.tensor(lab).reshape(1,).to(self.device).long()
        neg = torch.cat([pc0_embeds[:lab], pc0_embeds[lab+1:]], dim=0)
        pos = pc0_embeds[lab].expand(neg.shape[0], -1).contiguous()
        anc = pc1_embed.expand(neg.shape[0], -1).contiguous()
        loss = self.crit(anc, pos, neg)
        ind_pred = torch.argmin(torch.sum((pc0_embeds-pc1_embed)**2, dim=1))
        return loss, ind_pred
    
    def forward(self, pc0, pc1, quat_gt, lab, quats):
        pc0 = torch.tensor(pc0).to(self.device).float()
        pc1 = torch.tensor(pc1).to(self.device).float()
        fn_embed_pc0 = lambda pc_batch: self.pc_embed0(pc_batch)
        pc0_embeds = batch_run(pc0, fn_embed_pc0, batch_size=20)
        pc1_embed = self.pc_embed1(pc1.unsqueeze(0))
        if self.proj_head:
            pc1_embed = self.proj_head(pc1_embed)
        loss, ind_pred = self.calc_loss(pc0_embeds, pc1_embed, lab)
        quat_pred = quats[ind_pred]
        return loss, quat_pred, None
    
    def backward(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.train_step += 1
    
    def load(self, model_fname):
        checkpoint = torch.load(model_fname, map_location=self.device)
        self.pc_embed0.load_state_dict(checkpoint['pc_embed0'])
        if not self.share_weights:
            self.pc_embed1.load_state_dict(checkpoint['pc_embed1'])
        if self.proj_head:
            self.proj_head.load_state_dict(checkpoint['proj_head'])
        self.train_step = checkpoint['train_step']

    def save(self):
        state = {
                'pc_embed0': self.pc_embed0.state_dict(),
                'train_step': self.train_step}
        if not self.share_weights:
            state['pc_embed1'] = self.pc_embed1.state_dict()
        if self.proj_head:
            state['proj_head'] = self.proj_head.state_dict()
        path = self.logs_dir/f"{self.train_step}_rotator_match.pth"
        torch.save(state, path)
        return path

class VolMatchRotate(nn.Module):
    """Rotate module."""

    def __init__(self, rot_config, part_shape, kit_shape, np_part, np_kit, 
                logs_dir, logger, no_user_input, max_yaw_pitch,
                rotator_type, pointnet_type='pointnet2', double_run=False,
                temperature=1, dist_aware_loss=True, regression=False, 
                margin=1, share_wts=True, proj_head=False,
                lite=False):
        """Transport module for placing.

        Args:
            n_rotations: number of rotations of convolving kernel.
            crop_size: crop size around pick argmax used as convolving kernel.
            preprocess: function to preprocess input images.
        """
        super().__init__()

        self.device = get_device()
        # angles ranges
        self.no_user_input = no_user_input
        if no_user_input:
            self.vic_quats, self.vic_mats = sample_rot_roll(max_yaw_pitch/180*np.pi, 0.1, 10/180*np.pi)
        else:
            self.vic_quats, self.vic_mats = sample_rot(*rot_config)
        # print("Number of orientations: ", len(self.vic_quats))
        self.nR = len(self.vic_quats)
        self.part_shape = np.array(part_shape)
        self.kit_shape = np.array(kit_shape)
        self.np_part = np_part
        self.np_kit = np_kit

        # model
        # print(f'rotator_type: {rotator_type},  pointnet_type: {pointnet_type}, regression: {regression}')
        self.rotator_type = rotator_type
        self.pointnet_type = pointnet_type
        if self.rotator_type == 'classify':
            self.model = ClassifyPC(logs_dir, pointnet_type, dist_aware_loss, regression, temperature, lite)
        if self.rotator_type == 'match':
            self.model = MatchPC(logs_dir, pointnet_type, dist_aware_loss, margin, share_wts, proj_head)
        
        # logging
        self.logger = logger

    @staticmethod
    def from_cfg(vm_cfg: DictConfig, load_model: bool = False, log=True, model_path=None):
        p0_vol_shape = vm_cfg.p0_vol_shape
        p1_vol_shape = vm_cfg.p1_vol_shape
        no_user_input = vm_cfg.no_user_input
        max_yaw_pitch = float(vm_cfg.max_yaw_pitch)
        max_angle = vm_cfg.max_perturb_angle
        delta_angle = vm_cfg.delta_angle
        rot_config = (max_angle, delta_angle)
        dist_aware_loss = vm_cfg.dist_aware_loss
        regression = vm_cfg.regression
        double_run = vm_cfg.double_run
        pointnet_type = vm_cfg.pointnet_type
        rotator_type = vm_cfg.rotator_type
        np_part = int(vm_cfg.np_part)
        np_kit = int(vm_cfg.np_kit)
        temperature = float(vm_cfg.temperature)
        margin = float(vm_cfg.margin)
        share_wts = vm_cfg.share_wts
        proj_head = vm_cfg.proj_head
        lite = vm_cfg.lite
        logs_dir, logger = None, None
        if log:
            logs_dir = init_logs_dir(vm_cfg, f'{vm_cfg.vol_type}_rotate_full')
            logger = SummaryWriter(logs_dir /"tensorboard")
        model = VolMatchRotate(rot_config, p0_vol_shape, p1_vol_shape, np_part, np_kit,
                                logs_dir, logger, no_user_input, max_yaw_pitch,
                                rotator_type, pointnet_type, double_run,
                                temperature, dist_aware_loss, regression,
                                margin, share_wts, proj_head,
                                lite)
        if load_model:
            if model_path is None:
                model.load(Path(vm_cfg.rotator_path))
            else:
                model.load(Path(model_path))
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

    def forward(self, pc0_norm, pc1_norm, 
                vic_mats, vic_quats, quat_gt, concav_ori, syms, 
                sample=False, sample_num=40, calc_loss=True):
        # sample orientations if necessary, get gt orientation
        sample_inds = list(range(vic_mats.shape[0]))
        # get gt index
        lab = None
        if calc_loss:
            quat_dists = np.array([get_quat_diff_sym(quat_gt, quat, concav_ori, syms) for quat in vic_quats])
            rotation_index = np.argmin(quat_dists)
            lab = rotation_index
            if sample:
                sample_inds = random.sample(sample_inds[:rotation_index]+sample_inds[rotation_index+1:], sample_num-1)
                sample_inds.append(rotation_index)
                lab = sample_num-1
        rot_mats, rot_quats = vic_mats[sample_inds], vic_quats[sample_inds]
        # rotate point clouds
        pc0_rot = np.einsum('rij, jp -> rip', rot_mats, pc0_norm) 

        loss, quat_pred, aux = self.model.forward(pc0_rot, pc1_norm, quat_gt, lab, concav_ori, syms, rot_quats, calc_loss)
        return loss, quat_pred, aux

    def run(self, sample, training=False, log=True, calc_loss=True):
        self.train(training)
        # prepare data
        p0_vol, p1_vol, p1_coords, p1_coords_user, p1_ori, concav_ori, syms = sample.values()
        if torch.is_tensor(p0_vol):
            p0_vol = p0_vol.cpu().numpy()
        if torch.is_tensor(p1_vol):
            p1_vol = p1_vol.cpu().numpy()
        if torch.is_tensor(p1_coords):
            p1_coords = p1_coords.cpu().numpy()
        p0_vol, p1_vol = p0_vol[0], p1_vol[0] # assume batch_size=1
        p_gt = p1_coords[0] # assume batch_size=1
        concav_ori, syms = concav_ori[0].cpu().detach().numpy(), syms[0].numpy()
        if training: # perturb p_gt
            p_gt[0] += random.randint(-3,3)
            p_gt[1] += random.randint(-3,3)
            p_gt[2] += random.randint(-3,3)
        # crop kit volume and sample point clouds
        p1_vol_crop = center_crop(p1_vol, p_gt, self.kit_shape, tensor=False)
        pc0 = self.sample_pointcloud_from_tsdf(p0_vol, self.np_part)
        pc1 = self.sample_pointcloud_from_tsdf(p1_vol_crop, self.np_kit)
        pc0_norm, pc1_norm = self.normalize_pc(pc0, pc1)
        # get prediction and loss
        sample = training and self.pointnet_type == 'pointnet2'
        quat_gt = p1_ori[0].cpu().numpy() if p1_ori is not None else None # assume batch_size=1
        loss, quat_pred, aux = self.forward(pc0_norm, pc1_norm, self.vic_mats, self.vic_quats, 
                                            quat_gt, concav_ori, syms, 
                                            sample, sample_num=40, calc_loss=calc_loss)
        ori_diff = None
        if quat_gt is not None:
            ori_diff = get_quat_diff_sym(quat_gt, quat_pred, concav_ori, syms)
        # logging
        if log and calc_loss:
            print_str = f'ori_diff: {ori_diff:.3f} '
            if aux is not None:
                for k, v in aux.items():
                    print_str += f'{k}: {v:.3f} '
            # print(print_str)
            self.log(loss, ori_diff, aux)
        # update params
        if training and calc_loss:
            self.model.backward(loss)
        if loss is not None:
            loss = loss.item()
        return loss, p_gt, quat_pred, ori_diff
                
    def log(self, total_loss, ori_diff, aux):
        if self.logger == None:
            return
        log_dic = {
            'total_loss': total_loss,
            'ori_diff': ori_diff,
        }
        if aux is not None:
            for k, v in aux.items():
                log_dic[k] = v
                log_dic[k] = v
        for k, v in log_dic.items():
            self.logger.add_scalar(k, v, self.model.train_step)

    def load(self, model_fname):
        print(f'Loaded from {model_fname}')
        self.model.load(model_fname)

    def save(self):
        path = self.model.save()
        print(f"Model saved at path: {path}")
