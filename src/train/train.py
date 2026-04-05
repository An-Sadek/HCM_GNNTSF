from pathlib import Path
import sys

import yaml
import torch.nn as nn
from tqdm import tqdm
import torch.optim

if __package__:
    from ..data.dataloader import get_dataloader
    from ..model.stgnn import STGNN
else:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.data.dataloader import get_dataloader
    from src.model.stgnn import STGNN

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


def get_optimizer(name: str, **params):
    match name:
        case "adam": return torch.optim.Adam(**params)
        case "sgd": return torch.optim.SGD(**params)
        case _: assert False, "Không tìm thấy hàm tối ưu"

class Training:
    def __init__(self, model_architecture: str, model_name: str):
        assert model_architecture in ["STGNN", "Graph WaveNet"]
        self.model_architecture = model_architecture
        self.model_name = model_name
        self.model: nn.Module = nn.Module()

        self.config_dict = config["train"][model_architecture][model_name]
        self.optimizer_dict = self.config_dict["optimizer"]
        overall_dict = self.config_dict["overall"]

        # Khởi tạo các siêu tham số chung chung
        self.BATCH_SIZE = overall_dict["batch_size"]
        self.EPOCHS = overall_dict["epoch"]
        self.DROPOUT = overall_dict["dropout"]

        # Hàm tối ưu
        self.optimizer = get_optimizer(
            self.optimizer_dict["name"],
            **self.optimizer_dict["params"]
        )

    def train(self, train_loader, val_loader, test_loader):
        self.model.train()
        for epoch in tqdm(self.EPOCHS, desc=f"Đang train mô hình {self.model_architecture}({self.model_name})"):
            for data, forecast in tqdm(train_loader, desc="Batch size", leave=False):
                self.optimizer.zero_grad()
                y_hat = self.model(data)



class STGNNTraining(Training):
    def __init__(self, model_name: str):
        super().__init__("STGNN", model_name)

        # Khởi tạo siêu tham số của mô hình
        self.F = self.config_dict["F"]
        self.K = self.config_dict["K"]
        self.d = self.config_dict["d"]
        self.L = self.config_dict["L"]

        self.model = STGNN(self.F, self.K * self.d, self.L, self.d)

    
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloader()
