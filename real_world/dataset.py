from torch.utils.data import Dataset
import numpy as np
import sys
from pathlib import Path
root_path = Path(__file__).parent.parent.absolute()
# print("Adding to python path: ", root_path)
sys.path = [str(root_path)] + sys.path
import json

class REAL_DATASET(Dataset):
    def __init__(self, dataset_root: Path) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.dataset_root.mkdir(parents=True, exist_ok=True)

        self.datapoint_paths = [
            datapoint_path
            for datapoint_path in self.dataset_root.iterdir()
            if datapoint_path.is_dir() and not datapoint_path.name.startswith(".")
        ]
        self.datapoint_paths.sort(key=lambda x: int(str(x.name)))
        self.camera_pose = np.load(dataset_root / "camera_pose.npy")
        self.camera_depth_scale = np.load(
            dataset_root / "camera_depth_scale.npy")
        self.camera_color_intr = np.load(
            dataset_root / "camera_color_intr.npy")
        self.camera_depth_intr = np.load(dataset_root / "camera_depth_intr.npy")
        self.kit_bounds_mask = (np.load(dataset_root / "kit_bounds_mask.npy") <= 0)

    def __len__(self) -> int:
        return len(self.datapoint_paths)

    def __getitem__(self, idx: int, use_idx_as_datapoint_folder_name: bool):
        if use_idx_as_datapoint_folder_name:
            datapoint_path = self.dataset_root / f"{idx}"
        else:
            datapoint_path = self.datapoint_paths[idx]
        rgb = np.load(datapoint_path / "rgb.npy")
        d = np.load(datapoint_path / "d.npy")

        gt_mesh_paths = None
        gt_id_file = datapoint_path / "gt_ids.json"
        if gt_id_file.exists():
            with open(gt_id_file) as gt_ids_fp:
                gt_mesh_paths = json.load(gt_ids_fp)

        return rgb, d, gt_mesh_paths, datapoint_path
