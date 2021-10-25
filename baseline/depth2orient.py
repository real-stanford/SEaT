
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from learning.quat_rgs_loss import QuatRgsLoss
from learning.dataset import Depth2OrientDataset
from utils import  get_device, seed_all_int
from utils.rotation import get_quat_diff, get_quat_diff_sym
from evaluate.html_vis import html_visualize, visualize_helper

from os.path import join
from pathlib import Path


class ConvBNSELU(nn.Sequential):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, groups=1, bias=True, dilation=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNSELU, self).__init__(
                nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=bias,dilation=dilation),
                nn.BatchNorm2d(C_out),
                nn.SELU(inplace=True)
        )

class ConvBNReLU(nn.Sequential):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, groups=1, bias=True, dilation=1):
            padding = (kernel_size - 1) // 2
            super(ConvBNReLU, self).__init__(
                    nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=bias,dilation=dilation),
                    nn.BatchNorm2d(C_out),
                    nn.ReLU(inplace=True)
            )

class ResnetBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, bias=False):
        super(ResnetBasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.conv1 = ConvBNReLU(C_in=1,C_out=64,kernel_size=3,stride=2, bias=False) #SE3 is 4->64, 7x7, stride 2
        # self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        # self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv2 = ResnetBasicBlock(64,64,bias=False, stride=2)
    
    def forward(self, x):
        out = self.conv1(x)
        # out = self.pool1(out)
        out = self.conv2(out)
        return out

class ResNetSiameseNetwork(nn.Module):
    def __init__(self, split_resnet = True):
        super(ResNetSiameseNetwork, self).__init__()
        self.resnet = FeatureNet()
        if split_resnet:
            self.resnet2 = FeatureNet()
        self.split_resnet = split_resnet

        self.conv1 = ConvBNReLU(128,256,stride=2)
        self.conv2 = ResnetBasicBlock(256,256)
        self.conv3 = ConvBNReLU(256,256,stride=2)
        self.conv4 = ResnetBasicBlock(256,256)
        self.conv5 = ConvBNReLU(256,256,stride=2)
        self.conv6 = ResnetBasicBlock(256,256)
        self.pool = nn.AdaptiveAvgPool2d((4,4))
        self.final_fc = nn.Linear(4096,4)

    def forward(self, input1, input2):
        output1 = self.resnet(input1)
        output2 = self.resnet2(input2) if self.split_resnet else self.resnet(input2)
        output_concat = torch.cat((output1, output2), 1)

        output = self.conv1(output_concat)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = self.pool(output)
        output = output.reshape(output.shape[0],-1)
                
        output = self.final_fc(output)
        # print(output)
        output = F.normalize(output)
        return output

class Depth2Orient(nn.Module):
    def __init__(self):
        super(Depth2Orient, self).__init__()
        self.device = get_device()
        self.model = ResNetSiameseNetwork().to(self.device).float()
        self.optim = optim.Adam(self.model.parameters(), lr=2e-3)
        self.crit = QuatRgsLoss(w=1)
        self.logs_dir = "baseline/logs/depth2orient"
        self.logger = SummaryWriter(join(self.logs_dir, "tensorboard"))
        self.train_step = 0
        
    def run(self, sample, training=False, log=True):
        self.train(training)
        depth1, depth2, pc, gt_quat, concav_ori, syms = sample
        depth1 = depth1.to(self.device).unsqueeze(1).float()
        depth2 = depth2.to(self.device).unsqueeze(1).float()
        pc = pc.to(self.device).float()
        gt_quat = gt_quat.to(self.device).float()
        pred_quat = self.model(depth1, depth2)
        loss = self.crit(pred_quat, gt_quat, pc)
        pred_quat_np = pred_quat.detach().cpu().numpy()
        gt_quat_np = gt_quat.detach().cpu().numpy()
        concav_ori_np = concav_ori.detach().cpu().numpy()
        syms_np = syms.detach().cpu().numpy()

        diffs = np.array([
            get_quat_diff_sym(gt_quat_sp, pred_quat_sp, concav_ori_sp, syms_sp) \
            for gt_quat_sp, pred_quat_sp, concav_ori_sp, syms_sp \
            in zip(gt_quat_np, pred_quat_np, concav_ori_np, syms_np)
        ]) # deg
        # diffs = get_quat_diff(gt_quat_np, pred_quat_np) * 180 / np.pi

        if training:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.train_step += 1
        if log and training:
            self.logger.add_scalar('loss', loss, self.train_step)
            self.logger.add_scalar('diff', np.median(diffs), self.train_step)
            # print(f'step: {self.train_step}, loss: {loss:.3f}, diff: {np.median(diffs):.3f}')
        return loss.item(), pred_quat_np, diffs
        
    def load(self, path):
        print(f'Loaded from {path}')
        state = torch.load(path)
        self.model.load_state_dict(state['model'])

    def save(self, epoch):
        state = {'model': self.model.state_dict()}
        path = f'{self.logs_dir}/model_{epoch}.pth'
        torch.save(state, path)
        print(f"Model saved at path: {path}")

def test_run():
    model = Depth2Orient()

    depth1 = torch.rand(10,32,32)
    depth2 = torch.rand(10,32,32)
    pc = torch.rand(10, 3, 128)
    gt_quat = torch.rand(10, 4)
    concav_ori = torch.rand(10, 4)
    symmetry = torch.zeros(10, 3)-1
    sample = (depth1, depth2, pc, gt_quat, concav_ori, symmetry)

    for i in range(1):
        loss, pred, diffs = model.run(sample, training=True, log=True)
        print(loss)
        print(pred)
        print(diffs)

    model.save()
    model.load()

def train():
    seed_all_int(3)
    epoch_num = 800
    batch_size = 12
    save_interval = 8

    dataset = Depth2OrientDataset(Path('dataset/vol_match_abc/train'))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valset = Depth2OrientDataset(Path('dataset/vol_match_abc/val'))
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=1)

    model = Depth2Orient()
    pbar = tqdm(range(epoch_num), desc=f"Training depth2orient", dynamic_ncols=True)
    best_val_diff = np.Inf
    for epoch in pbar:
        total_loss, total, all_diffs = 0, 0, []
        for batch in dataloader:
            loss, _, diffs = model.run(batch, training=True, log=True)
            total_loss += loss * batch[0].shape[0]
            total += batch[0].shape[0]
            all_diffs += list(diffs)
        if (epoch+1) % save_interval == 0:
            all_val_diffs = []
            for batch in valloader:
                _, _, diffs = model.run(batch, training=True, log=True)
                all_val_diffs += list(diffs)
            med = np.median(all_val_diffs)
            best_val_diff = min(best_val_diff, med)
            model.save(epoch)
            pbar.set_postfix(mean_loss = f"{total_loss/total:.2f}",
                             med_diff = f"{np.median(np.array(all_diffs)):.2f}",
                             val_diff = f"{med:.2f}",
                             best_diff = f"{best_val_diff:.2f}")

def evaluate():
    seed_all_int(4)
    batch_size = 20

    dataset = Depth2OrientDataset(Path('dataset/vol_match_abc/val'), augment=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    model = Depth2Orient()
    model.load('logs/depth2orient/model_703.pth')
    outputdir = Path('eval_d2o')
    outputdir.mkdir(exist_ok=True)
    cnt = 0
    all_diffs = []
    tasks = []
    for batch in dataloader:
        with torch.no_grad():
            _, preds, diffs = model.run(batch, training=False, log=False)
        part_img, kit_img, pc, ori, concav_ori, symmetry = batch
        for i, pred in enumerate(preds):
            dump_paths = Depth2OrientDataset.visualize_depth2orient(outputdir/str(cnt), 
                            part_img[i], kit_img[i], 
                            pc[i], ori[i], concav_ori[i], symmetry[i], 
                            ori_pred=pred)
            tasks.append(dump_paths)
            cnt += 1
        all_diffs += list(diffs)
    np.save(outputdir / 'diffs.npy', np.array(all_diffs))
    cols = ["part_img", "kit_img", "part_img_rot", "overlay", "ori", "symmetry", "part_img_rot_pred", "overlay_pred", "ori_pred", "ori_diff"]
    visualize_helper(tasks, outputdir, cols)
    all_diffs = np.array(all_diffs)
    print(f'med_diff: {np.median(all_diffs):.3f}, mean_diff: {np.mean(all_diffs):.3f}, max_diff: {np.max(all_diffs):.3f}')

if __name__ == "__main__":
    evaluate()