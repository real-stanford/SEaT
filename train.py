
from baseline.transportnet import Transportnet
from learning.vol_match_transport import VolMatchTransport
from learning.vol_match_rotate import VolMatchRotate
import hydra
from omegaconf import DictConfig
from learning.dataset import SceneDatasetShapeCompletion, SceneDatasetShapeCompletionSnap, TNDataset, VolMatchDataset, SceneDatasetMaskRCNN
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from learning.shape_comp_model import ShapeCompModel
from learning.shape_comp_model_new import ShapeCompModelNew
from tqdm import tqdm
from pathlib import Path
import numpy as np
from utils import get_device, init_logs_dir, next_loop_iter, seed_all_int, calcMIoU
from learning.seg import get_instance_segmentation_model, get_transform
from vision_utils import utils
from vision_utils.engine import train_one_epoch, evaluate

def train_shape_completion(cfg: DictConfig):
    # Load dataset 
    scene_type = cfg.train.scene_type

    dataset = SceneDatasetShapeCompletion(Path(cfg.train.dataset_path) / scene_type / "train_sc", scene_type)
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.ray.num_cpus)
    valset = SceneDatasetShapeCompletionSnap(Path('dataset/vol_match_abc/val'), scene_type)
    val_dataloader = DataLoader(valset, batch_size=cfg.train.batch_size, num_workers=cfg.evaluate.num_workers, shuffle=False)

    device = get_device()
    vol_shape = cfg.env.obj_vol_shape if scene_type=='object' else cfg.env.kit_vol_shape
    sc_model = ShapeCompModelNew(vol_shape).to(device)
    optim = Adam(sc_model.parameters())
    criterion = MSELoss()

    logs_dir = Path(cfg.train.log_path)
    logs_dir.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(range(cfg.train.epochs), desc="Training SC", dynamic_ncols=True)
    def train_sc_epoch(dl: DataLoader, train:bool=True):
        total_IoU, cnt = 0, 0
        for inps, targets in dl:
            inps, targets = inps.to(device, dtype=torch.float), targets.to(device, dtype=torch.float)
            sc_model.train(train)
            if train:
                preds = sc_model(inps)
            else:
                with torch.no_grad():
                    preds = sc_model(inps)
            cnt += inps.shape[0]
            total_IoU += inps.shape[0] * calcMIoU(preds, targets)
            if train:
                loss = criterion(preds, targets)
                optim.zero_grad()
                loss.backward()
                optim.step()
        mIoU = total_IoU/cnt
        return mIoU

    for epoch in pbar:
        mIoU = train_sc_epoch(dataloader)
        if (epoch + 1) % cfg.train.save_model_every == 0 or (epoch + 1) == cfg.train.epochs:
            model_path = logs_dir / f"sc_{epoch+1}.pth"
            torch.save(sc_model, model_path)
            mIoU_val = train_sc_epoch(val_dataloader, train=False)
            print(f"Model saved: {model_path}; train mIoU: {mIoU:.2f}; val mIoU: {mIoU_val:.2f}")
        pbar.set_postfix(mean_mIoU = f"{mIoU:.2f}")
    pbar.close()

def train_seg(cfg: DictConfig):
    device = get_device()
    logs_dir = init_logs_dir(cfg, f'seg')
    print(f"Saving logs in {logs_dir}")
    dataset_root = Path(cfg.train.dataset_path)
    use_depth = cfg.train.use_depth
    normalize_depth = cfg.train.normalize_depth

    train_transforms = get_transform(True, use_depth, normalize_depth)
    train_dataset = SceneDatasetMaskRCNN(
        dataset_root=dataset_root / "train", use_depth=use_depth, transforms=train_transforms)
    train_dataset.print_statistics("Train|")
    data_loader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers = cfg.train.num_cpus,
        collate_fn=utils.collate_fn)

    val_transforms = get_transform(False, use_depth, normalize_depth)
    val_dataset = SceneDatasetMaskRCNN(
        dataset_root=dataset_root / "val", use_depth=use_depth, transforms=val_transforms)
    val_dataset.print_statistics("Val|")
    data_loader_val = DataLoader(
        val_dataset, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers = cfg.train.num_cpus,
        collate_fn=utils.collate_fn)

    num_classes = 3

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = cfg.train.epochs
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1} / {num_epochs}")
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_val, device=device)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            torch.save(model, f"{logs_dir}/{epoch}.pth")


def train_tn(cfg: DictConfig):
    dataset = TNDataset.from_cfg(cfg.train, 'train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    dataloader_iter = iter(dataloader)
    model = Transportnet.from_cfg(cfg.train)
    valset = TNDataset.from_cfg(cfg.train, 'val')
    valset_iter = iter(valset)

    train_steps = cfg.train.train_steps
    save_interval = cfg.train.save_interval

    num_samples = 20

    pbar = tqdm(range(train_steps), desc=f"Training tn", dynamic_ncols=True)
    for epoch in pbar:
        dataloader_iter, next_batch = next_loop_iter(dataloader_iter, dataloader)
        sample = (item[0].numpy() for item in next_batch)
        model.run(sample, training=True)
        if (epoch+1) % save_interval == 0:
            total_diff_ori, total_diff_pos = 0, 0
            for _ in range(num_samples):
                valset_iter, sample = next_loop_iter(valset_iter, valset)
                with torch.no_grad():
                    losses, preds, (pos_diff, ori_diff) = model.run(sample, training=False)
                    total_diff_pos += pos_diff
                    total_diff_ori += ori_diff
            pbar.set_postfix(pos_diff=f'{total_diff_pos/num_samples:.2f}',
                             ori_diff=f'{total_diff_ori/num_samples:.2f}')
            model.save()

def train_vol_match(cfg: DictConfig):
    name = cfg.train.name
    vm_cfg = cfg.vol_match_6DoF
    no_user_input = vm_cfg.no_user_input

    if name == "vol_match_transport":
        vol_matcher = VolMatchTransport.from_cfg(vm_cfg, cfg.env.voxel_size, vm_cfg.load_model) 
        train_steps = 20000
        save_interval = 200
        batch_size = 1 if no_user_input else 4
        num_samples = 10
    elif name == "vol_match_rotate":
        vol_matcher = VolMatchRotate.from_cfg(vm_cfg, vm_cfg.load_model) 
        train_steps = 150000
        save_interval = 500
        batch_size = 1
        num_samples = 100

    dataset = VolMatchDataset.from_cfg(cfg, Path(vm_cfg.dataset_path) / 'train', vol_type=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    dataloader_iter = iter(dataloader)
    valset = VolMatchDataset.from_cfg(cfg, Path(vm_cfg.dataset_path) / 'val', vol_type=None)
    valset_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=1)
    valset_loader_iter = iter(valset_loader)
    if no_user_input:
        num_samples = len(valset)

    pbar = tqdm(range(train_steps), desc=f"Training {name}", dynamic_ncols=True)
    losses, diffs, best_val_diff = [], [], np.Inf
    for epoch in pbar:
        dataloader_iter, next_batch = next_loop_iter(dataloader_iter, dataloader)
        loss, _, _, diff = vol_matcher.run(next_batch, training=True, log=True, calc_loss=True)
        if loss is not None:
            losses.append(loss)
            diffs.append(diff)
        if (epoch+1) % save_interval == 0:
            vol_matcher.save()
            val_diffs = []
            for _ in range(num_samples):
                valset_loader_iter, next_batch = next_loop_iter(valset_loader_iter, valset_loader)
                with torch.no_grad():
                    _, _, pred, diff = vol_matcher.run(next_batch, training=False, log=False, calc_loss=True)
                if diff is not None:
                    val_diffs.append(diff)
            val_med = np.median(np.array(val_diffs))
            if  val_med < best_val_diff:
                print(f'New SoTA: {val_med} at {epoch+1}.')
                best_val_diff = val_med
            pbar.set_postfix(mean_loss = f"{np.mean(np.array(losses)):.2f}",
                            med_diff = f"{np.median(np.array(diffs)):.2f}",
                            val_diff = f"{val_med:.2f}",
                            best_diff = f"{best_val_diff:.2f}")  
            losses, diffs = [], []

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Add save directory in hydra config    
    seed_all_int(cfg.seeds.train)
    # init_logs_dir(cfg, cfg.train.name)
    if cfg.train.name == "shape_completion":
        train_shape_completion(cfg)
    elif cfg.train.name == "seg":
        train_seg(cfg)
    elif cfg.train.name == "tn":
        train_tn(cfg)
    elif cfg.train.name in ["vol_match_transport", "vol_match_rotate", "pointnet_regressor"]:
        train_vol_match(cfg)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
