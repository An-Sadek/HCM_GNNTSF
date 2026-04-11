from pathlib import Path

import h5py
import torch
import yaml
from torch.utils.data import DataLoader, Dataset


torch.manual_seed(42)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

path_config = config["data"]["path"]
DYNAMIC_PATH = path_config["dynamic"]
STATIC_PATH = path_config["static"]
GRAPH_PATH = path_config["graph"]

split_ratio = config["data"]["split_ratio"]
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = split_ratio


def _load_h5_tensor(path: str, key: str) -> torch.Tensor:
    with h5py.File(path, "r") as f:
        if key not in f:
            available_keys = ", ".join(sorted(f.keys()))
            raise KeyError(f"'{key}' not found in {path}. Available keys: {available_keys}")
        return torch.from_numpy(f[key][:])


def get_dynamic_feature_dim(split: str = "train") -> int:
    with h5py.File(DYNAMIC_PATH, "r") as f:
        if split not in f:
            available_splits = ", ".join(sorted(f.keys()))
            raise KeyError(f"'{split}' not found in {DYNAMIC_PATH}. Available splits: {available_splits}")
        return int(f[split]["X"].shape[-1])


static = _load_h5_tensor(STATIC_PATH, "static")
graph = _load_h5_tensor(GRAPH_PATH, "edge_index")


class HCM_Dataset(Dataset):
    def __init__(self, h5_path: str, split: str):
        self.h5_path = h5_path
        self.split = split
        with h5py.File(self.h5_path, "r") as f:
            if self.split not in f:
                available_splits = ", ".join(sorted(f.keys()))
                raise KeyError(
                    f"'{self.split}' not found in {self.h5_path}. "
                    f"Available splits: {available_splits}"
                )
            self.length = f[self.split]["X"].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        with h5py.File(self.h5_path, "r") as f:
            split_group = f[self.split]
            X = torch.from_numpy(split_group["X"][idx]).float()
            y = torch.from_numpy(split_group["y"][idx]).float()
        return X, y


train_dataset = HCM_Dataset(h5_path=DYNAMIC_PATH, split="train")
val_dataset = HCM_Dataset(h5_path=DYNAMIC_PATH, split="val")
test_dataset = HCM_Dataset(h5_path=DYNAMIC_PATH, split="test")


def get_dataloader(batch_size: int = 32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("X[0] shape:")
    print(train_dataset[0][0].shape)
    print("\ny[0] shape:")
    print(train_dataset[0][1].shape)
    print("\nStatic shape:")
    print(static.shape)
    print("\nGraph shape:")
    print(graph.shape)
    print("\nDataset split lengths:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")
