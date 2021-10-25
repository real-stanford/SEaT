import torch
from torch import nn
from utils import get_device
from utils.rotation import torch_quat_to_mat

class QuatRgsLoss(torch.nn.Module):
    """
    Shape-Match Loss function.
    Based on PoseCNN
    """

    def __init__(self, w=10):
        super(QuatRgsLoss, self).__init__()
        self.device = get_device()
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.w = w

    @staticmethod
    def compute_pc_dist(x, y):
        # code from https://github.com/YanWei123/Pytorch-implementation-of-FoldingNet-encoder-and-decoder-with-graph-pooling-covariance-add-quanti
        """x is ground truth point cloud, y is rotated point cloud"""
        x_size = x.size() # b, 3, np1
        y_size = y.size() # b, 3, np2
        x = x.unsqueeze(-1).expand(-1,-1,-1,y_size[-1]).contiguous() # b, 3, np1, np2
        y = y.unsqueeze(-2).expand(-1,-1,x_size[-1],-1).contiguous() # b, 3, np1, np2
        l2 = torch.pow(x-y, 2).sum(1) # b, np1, np2
        min_l2, _ = torch.min(l2, 1) # b, np2
        chamfer_distance = torch.mean(torch.mean(min_l2, 1))
        if torch.isnan(chamfer_distance):
            return 0
        return chamfer_distance

    def forward(self, predquat, gtquat, points):
        """
        predquat: (b, 4) or (4,)
        gtquat: (b, 4) or (4,)
        points: (b, 3, np) or (3, np)
        """
        assert predquat.shape == gtquat.shape, f'shape mismatch: pred is {predquat.shape}, gt is {gtquat.shape}'
        if len(gtquat.shape)==1:
            predquat = predquat.unsqueeze(0)
            gtquat = gtquat.unsqueeze(0)
            points = points.unsqueeze(0)
        if len(points.shape) == 2:
            points = points.expand(predquat.shape[0], -1, -1)

        assert gtquat.shape[1]==4, 'not quaternions'
        assert points.shape[1]==3, 'expect pints to have 3 coordinates'

        cos_loss = self.cos_loss(predquat, gtquat, torch.ones(predquat.shape[0]).to(self.device))

        predrot = torch_quat_to_mat(predquat, self.device).float()
        gtrot = torch_quat_to_mat(gtquat, self.device).float()
        predpts = torch.bmm(predrot, points)
        gtpts = torch.bmm(gtrot, points)
        pt_loss = self.compute_pc_dist(gtpts, predpts)
        return cos_loss

if __name__ == "__main__":
    loss = QuatRgsLoss()

    pred = torch.rand(4)
    gt = torch.rand(4)
    points = torch.rand(3, 1024)
    out = loss(pred, gt, points)
    print(out)

    pred = torch.rand(2,4)
    gt = torch.rand(2,4)
    points = torch.rand(2, 3, 1024)
    out = loss(pred, gt, points)
    print(out)
