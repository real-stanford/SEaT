from baseline.resnet import ResNet43_8s
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
import PIL

from utils import get_device
from utils.ravenutils import sample_distribution
from os.path import join
from torch.utils.tensorboard import SummaryWriter

class Matching(nn.Module):
    """Matching module."""
    def __init__(self, descriptor_dim, num_rotations):
        super(Matching, self).__init__()

        self.n_rotations = num_rotations
        self.descriptor_dim = descriptor_dim

        self.model = ResNet43_8s(6, self.descriptor_dim)
        self.optim = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        self.padding = (80,)*4
        self.pad_size = [80, 80]

        self.margin = 1
        self.num_samples = 100

    def forward(self, in_img):
        shape = in_img.shape
        in_tens = nn.functional.pad(in_img, self.padding)
        # Rotate angles.
        angles = [i*360/self.n_rotations for i in range(self.n_rotations)]
        # Forward pass.
        logits = ()
        for angle in angles:
            rotated_tensor = transforms.functional.affine(in_tens, angle=angle, translate=[0,0], 
                                                          scale=1, shear=[0,0], resample=PIL.Image.NEAREST)
            rotated_out = self.model(rotated_tensor)
            out = transforms.functional.affine(rotated_out, angle= -angle, translate=[0,0], 
                                               scale=1, shear=[0,0], resample=PIL.Image.NEAREST)
            logits += (out,)
        logits = torch.cat(logits, dim=0)
        hs, he = self.pad_size[0], self.pad_size[0]+shape[2]
        ws, we = self.pad_size[1], self.pad_size[1]+shape[3]
        output = logits[:, :, hs:he, ws:we]
        output = output.permute(0,2,3,1) # (1, H, W, descriptor_size)
        return output

    def get_pred(self, output, p, q):
        p_descriptor = output[0, p[0], p[1], :]
        q_descriptors = output[:, q[0], q[1], :]
        dists = torch.pow(p_descriptor.view(1,-1)-q_descriptors, 2)
        itheta_pred = torch.argmin(dists)
        theta_pred = itheta_pred / self.n_rotations * 2 * np.pi
        return theta_pred.item()

    def get_loss(self, output, p, q, theta):
        
        itheta = theta / (2 * np.pi / self.n_rotations)
        itheta = np.int64(np.round(itheta)) % self.n_rotations

        # Positives.
        p_descriptor = output[0, p[0], p[1], :]
        q_descriptor = output[itheta, q[0], q[1], :]
        pos_dist = torch.pow(p_descriptor - q_descriptor, 2).sum()
        
        # Negatives.
        sample_map = np.zeros(output.shape[0:3])
        sample_map[0, p[0], p[1]] = 1
        sample_map[itheta, q[0], q[1]] = 1
        sample_map = (1-sample_map).reshape(-1, )
        inds_neg = sample_distribution(sample_map, self.num_samples)
        output = output.reshape(np.prod(output.shape[:3]), output.shape[3])
        neg_descriptors = output[inds_neg, :] # n_sample, n_embed
        neg_dists = torch.pow(p_descriptor.reshape(1, -1)-neg_descriptors, 2).sum(1)
        neg_dists_margin, _ = torch.max(self.margin-neg_dists, 0)
        neg_dist_mean = neg_dists_margin.mean()

        loss = pos_dist + neg_dist_mean
        return loss

    def run(self, in_img, p, q, theta, train=True):
        self.train(train)
        output = self.forward(in_img)
        loss = self.get_loss(output, p, q, theta)
        pred = self.get_pred(output, p, q)

        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        return loss.item(), pred

class Attention(nn.Module):
    """Attention module."""

    def __init__(self):
        super(Attention, self).__init__()
        self.model = ResNet43_8s(6, 1)
        self.optim = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()

    def run(self, in_img, p, train=True):
        self.train(train)
        logits = self.model(in_img).squeeze()
        # Get label.
        label_size = in_img.shape[2:4]
        label = np.zeros(label_size)
        label[p[0], p[1]] = 1
        label = label.reshape(-1,)
        label = np.where(label==1)[0][0]
        label = torch.tensor(label, dtype=torch.int64).reshape(1,)
        # Get loss and pred.
        loss = self.criterion(logits.view(1, -1), label)
        pred = np.unravel_index(torch.argmax(logits), logits.shape)

        # Backpropagate
        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        return loss.item(), pred

class Form2Fit(nn.Module):
    def __init__(self):
        super(Form2Fit, self).__init__()
        self.device = get_device()
        self.logs_dir = "logs/form2fit/"
        self.logger = SummaryWriter(join(self.logs_dir, "tensorboard"))
        self.train_step = 0

        self.num_rotations = 24
        self.descriptor_dim = 16

        self.pick_model = Attention()
        self.place_model = Attention()
        self.match_model = Matching(self.descriptor_dim, self.num_rotations)
    
    def run(self, sample, training=False, log=True):
        self.train(training)
        img, p0, p1, theta_gt = sample 
        img = img.to(self.device).unsqueeze(0).float() # (1, 6, h, w)

        # Compute training loss.
        loss0, pred_p = self.pick_model.run(img, p0, training)
        loss1, pred_q = self.place_model.run(img, p1, training)
        loss2, pred_theta = self.match_model.run(img, p0, p1, theta_gt, training)

        if training:
            self.train_step += 1

        if log:
            self.logger.add_scalar('pick', loss0, self.train_step)
            self.logger.add_scalar('place',loss1, self.train_step)
            self.logger.add_scalar('match', loss2, self.train_step)
        
        return (loss0, loss1, loss2), (pred_p, pred_q, pred_theta)
        
    def load(self):
        path = f'{self.logs_dir}/model.pth'
        print(f'Loaded from {path}')
        state = torch.load(path)
        self.pick_model.load_state_dict(state['pick_model'])
        self.place_model.load_state_dict(state['place_model'])
        self.match_model.load_state_dict(state['match_model'])

    def save(self):
        state = {'pick_model': self.pick_model.state_dict(),
                 'place_model': self.place_model.state_dict(),
                 'match_model': self.match_model.state_dict(),
                }
        path = f'{self.logs_dir}/model.pth'
        torch.save(state, path)
        print(f"Model saved at path: {path}")

if __name__ == "__main__":
    model = Form2Fit()

    img = torch.rand(6,16,16)
    p0 = [5,5]
    p1 = [1,1]
    theta = np.random.random()
    sample = (img, p0, p1, theta)

    for i in range(5):
        loss, pred = model.run(sample, training=True, log=True)
        print(loss)
        print(pred)

    model.save()
    model.load()