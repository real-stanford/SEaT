import ray
from utils.ravenutils import np_unknown_cat, torch_unknown_cat
import torch
from torch import nn
from torch.nn import functional as F, parameter
from utils import get_device, get_ray_fn, init_ray, rotate_tsdf, sample_quaternions, distance_quaternion
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from scipy.ndimage import rotate
import pybullet as p
from tqdm import tqdm
import random
from tensorboardX import SummaryWriter
import time
import h5py


class VolEncoder(nn.Module):
    """
        in_shape: (B, C, T, H, W)    
        out_shape: (B, C, T // 2, H // 2, W // 2)
    """
    def __init__(self) -> None:
        super().__init__()

        self.conv0 = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm3d(1)

        self.conv1 = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(1)

        self.conv2 = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(1)

        self.conv3 = nn.Conv3d(1, 1, kernel_size=2, stride=2, padding=0)
        self.bn3 = nn.BatchNorm3d(1)

    def forward(self, x):
        x = x + F.leaky_relu(self.bn0(self.conv0(x)))
        x = x + F.leaky_relu(self.bn1(self.conv1(x)))
        x = x + F.leaky_relu(self.bn2(self.conv2(x)))
        x = torch.tanh(self.bn3(self.conv3(x)))
        return x


class MatchPredictor(nn.Module):
    def __init__(self, n_rotations) -> None:
        super().__init__()

        self.conv0 = nn.Conv3d(n_rotations, n_rotations + 1,
                               kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm3d(n_rotations + 1)

        self.conv1 = nn.Conv3d(n_rotations + 1, n_rotations + 1,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(n_rotations + 1)

        self.conv2 = nn.Conv3d(n_rotations + 1, n_rotations + 1,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(n_rotations + 1)

        self.conv3 = nn.Conv3d(n_rotations + 1, n_rotations + 1,
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(n_rotations + 1)

        self.conv4 = nn.Conv3d(n_rotations + 1, n_rotations + 1,
                               kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(n_rotations + 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn0(self.conv0(x)))
        x = x + F.leaky_relu(self.bn1(self.conv1(x)))
        x = x + F.leaky_relu(self.bn2(self.conv2(x)))
        x = x + F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        return x


class VolMatcher:
    def __init__(self, max_pert_theta, max_pert_rot, n_z, n_phi, n_rot, use_neg_label_sampling: bool,
        n_neg_label_samples: int, model_path: Path, logs_dir: Path) -> None:
        self.max_pert_theta = max_pert_theta
        self.max_pert_rot = max_pert_rot
        self.n_z = n_z
        self.n_phi = n_phi
        self.n_rot = n_rot
        self.n_quat = (self.n_z * self.n_phi + 1) * self.n_rot
        self.device = get_device()
        self.p0_encoder = VolEncoder().to(self.device)
        self.p1_encoder = VolEncoder().to(self.device)
        self.match_predictor = MatchPredictor(self.n_quat).to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.optim = torch.optim.Adam(
            params=list(self.p0_encoder.parameters(
            )) + list(self.p1_encoder.parameters()) + list(self.match_predictor.parameters()),
        )
        self.use_neg_loss_sampling = use_neg_label_sampling
        self.n_neg_label_samples = n_neg_label_samples
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True, parents=True)
        self.writer = SummaryWriter(logdir=self.logs_dir, flush_secs=30)

        if model_path is not None:
            self.load(model_path)
        
        self.train_step = 0
    
    @staticmethod
    def from_cfg(vm_cfg: DictConfig, logs_dir: str, load_model: bool = False):
        model_path = Path(vm_cfg.model_path) if load_model else None
        max_pert_theta = vm_cfg.max_pert_theta
        max_pert_rot = vm_cfg.max_pert_rot
        n_z = vm_cfg.n_z
        n_phi = vm_cfg.n_phi
        n_rot = vm_cfg.n_rot
        n_neg_label_samples = vm_cfg.n_neg_label_samples
        use_neg_label_sampling = n_neg_label_samples != -1
        return VolMatcher(max_pert_theta, max_pert_rot, n_z, n_phi, n_rot, use_neg_label_sampling,
                          n_neg_label_samples, model_path, logs_dir)

    def forward(self, p0_vol: np.ndarray, p1_vol: np.ndarray, p0_vols_rotate: np.ndarray): # , hdf_path
        # p0_vol: B, t, h, w
        # p1_vol: B, T, H, W
        # p0_vols_rotate: (B, R, t, h, w)
        # p1_vic_quats: B, R, 4
        batch_size = len(p0_vol)
        p1_vol = torch.tensor(p1_vol, dtype=torch.float,
                              device=self.device).unsqueeze(dim=1)  # (B, 1, T, H, W)

        # Just do it one by one for now
        conv_outputs = None
        p1_encodings = self.p1_encoder(p1_vol)  # (B, 1, T', H', W') 
        p1_THW = np.array(p1_encodings.shape)[2:]
        for batch_i in range(batch_size):
            p0_vols_rotate_batch = torch.tensor(p0_vols_rotate[batch_i], dtype=torch.float, device=self.device).unsqueeze(dim=1)  # (R, 1, t, h, w)
            p0_encodings = self.p0_encoder(p0_vols_rotate_batch)  # (R, 1, t, h, w)
            # Choose padding such that output shape is same as p1_vol shape
            padding_size = np.ceil(
                (np.array(p0_encodings.shape)[2:] - 1) / 2).astype(np.int)
            conv_output = F.conv3d(
                p1_encodings[batch_i].unsqueeze(dim=0),  # (1, 1, T, H, W)
                p0_encodings,  # (R, 1, t, h, w)
                padding=tuple(padding_size),
                stride=1
            )  # (1, R, T, H, W)
            conv_output_THW = np.array(conv_output.shape)[2:]
            if (conv_output_THW != p1_THW).any():
                # Make sure that output shape only exceeds by 1 unit
                indices = conv_output_THW != p1_THW
                assert (conv_output_THW[indices] - 1 == p1_THW[indices]).all()
                # We will simply ignore that exceeding dimension (which is conecptually correct as well)
                conv_output = conv_output[:, :, :p1_THW[0], :p1_THW[1], :p1_THW[2]]
            conv_outputs = torch_unknown_cat(conv_outputs, conv_output[0])
        return self.match_predictor(conv_outputs)  # (B, R, T', H', W')

    def prepare_train(self, train:bool):
        if train:
            self.p0_encoder.train()
            self.p1_encoder.train()
            self.match_predictor.train()
        else:
            self.p0_encoder.eval()
            self.p1_encoder.eval()
            self.match_predictor.eval()

    def run(self, sample, training: bool, logger=None):
        """
            p0_vol: B, t, h, w
            p1_vol: B, T, H, W
            p1_coords: B, 3
            p1_ori: B, 4
            p0_vols_rotated: B, R, t, h, w
        """
        since = time.time()
        self.prepare_train(training)
        p0_vol, p1_vol, p1_coords, p1_ori, p0_vols_rotated, p1_vic_quats = sample
        batch_size = len(p1_ori)

        pred_match_prob = self.forward(p0_vol, p1_vol, p0_vols_rotated)  # (B, R, T', H', W')

        target = torch.ones((batch_size, *pred_match_prob.shape[2:]), dtype=torch.long, device=self.device)  # (B, T', H', W')
        target *= self.n_quat
        for batch_i in range(batch_size):
            quat_distances = np.array([distance_quaternion(p1_quat, p1_ori[batch_i]) for p1_quat in p1_vic_quats[batch_i]])
            rotation_index = np.argmin(quat_distances)
            target[batch_i, p1_coords[batch_i][0] // 2, p1_coords[batch_i][1] // 2, p1_coords[batch_i][2] // 2] = rotation_index
        raw_loss = self.criterion(pred_match_prob, target)  # (B, T', H', W')

        if self.use_neg_loss_sampling:
            neg_indices = random.sample(list(zip(*np.where(target.detach().cpu().numpy() == self.n_quat))), self.n_neg_label_samples)
            pos_indices = list(zip(*np.where(target.detach().cpu().numpy() != self.n_quat)))
            indices = neg_indices + pos_indices
            loss = torch.tensor(0, device=self.device, dtype=torch.float)
            for ind in indices:
                loss = loss + raw_loss[ind]
            loss = loss / len(indices)
        else:
            loss = raw_loss.mean()

        if training:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        pred_coords = None
        pred_ori = None
        valid_ori_indices = np.ones(batch_size, dtype=np.bool)  # The network can predict that no orientation matches found.
        pred_match_prob = pred_match_prob.detach().cpu().numpy()
        for batch_i in range(batch_size):  # XXX: Convert this loop to batch operations. It can be done
            pred_match_prob_batch = pred_match_prob[batch_i]  # (R, H, W, D)
            indices = np.array(np.where(pred_match_prob_batch == np.max(pred_match_prob_batch))).squeeze()
            if indices[0] == self.n_quat:
                valid_ori_indices[batch_i] = False
                pred_ori = np_unknown_cat(pred_ori, np.zeros((4,)))
            else:
                pred_ori = np_unknown_cat(pred_ori, p1_vic_quats[batch_i][indices[0]])
            pred_coords = np_unknown_cat(pred_coords, indices[1:] * 2)

        self.writer.add_scalar(
            "CELoss",
            loss.item(),
            self.train_step
        )
        dist_error = np.linalg.norm(pred_coords - p1_coords.detach().cpu().numpy(), axis=1).mean()
        self.writer.add_scalar(
            "Pred_coordinate_euclidean_distance_from_GT_in_voxels",
            dist_error,
            self.train_step
        )
        quat_error = distance_quaternion(pred_ori[valid_ori_indices], p1_ori.detach().cpu().numpy()[valid_ori_indices]).mean()
        self.writer.add_scalar(
            "Pred_ori_quaternion_distance_from_GT",
            quat_error,
            self.train_step
        )
        print(f"Train: {self.train_step + 1}| celoss = {loss.item():.3f}; de: {dist_error:.3f}; qe: {quat_error:.3f} (in {time.time() - since:.1f}s)")
        
        self.train_step += 1
        return loss.item(), pred_coords, pred_ori, valid_ori_indices
        
    def save(self, epoch=0):
        """Save models."""
        print(f"Model saved at path: {self.logs_dir} / {epoch}")
        torch.save(self.p0_encoder, self.logs_dir / f"{epoch}_p0_encoder.pth")
        torch.save(self.p1_encoder, self.logs_dir / f"{epoch}_p1_encoder.pth")
        torch.save(self.match_predictor, self.logs_dir / f"{epoch}_match_predictor.pth")

    def load(self, model_path:Path):
        model_dir = model_path.parent
        epoch = model_path.stem
        self.p0_encoder = torch.load(model_dir / f"{epoch}_p0_encoder.pth")
        self.p1_encoder = torch.load(model_dir / f"{epoch}_p1_encoder.pth")
        self.match_predictor = torch.load(model_dir / f"{epoch}_match_predictor.pth")
        print("VolMatcher models loaded from path: ", model_path)
