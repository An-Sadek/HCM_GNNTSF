# BEGIN === Khâu chuẩn bị
# Thư viện
import argparse
import os.path
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim
import yaml
from tqdm import tqdm

# Load package khác
if __package__:
    from ..data.dataloader import get_dataloader
    from ..model.stgnn import STGNN
else:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.data.dataloader import get_dataloader
    from src.model.stgnn import STGNN


# Load config và định nghĩa hằng
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Tạo thư mục result
RESULT_DIR = config["data"]["path"]["result"]
if not os.path.exists(RESULT_DIR):
    print("Không tìm thấy thư mục kết quả, tạo mới")
    os.mkdir(RESULT_DIR)
else:
    print("Xác nhận thư mục kết quả")
# === END


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, null_val: float = -1.0) -> torch.Tensor:
    mask = (target != null_val).float()
    mask = mask / torch.clamp(mask.mean(), min=1e-6)
    loss = torch.abs(prediction - target) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss.mean()


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor, null_val: float = -1.0):
    mask = target != null_val
    valid_pred = prediction[mask]
    valid_target = target[mask]

    if valid_target.numel() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}

    mae = torch.mean(torch.abs(valid_pred - valid_target)).item()
    rmse = torch.sqrt(torch.mean((valid_pred - valid_target) ** 2)).item()
    denom = torch.clamp(valid_target.abs(), min=1e-6)
    mape = torch.mean(torch.abs((valid_pred - valid_target) / denom)).item()
    return {"mae": mae, "rmse": rmse, "mape": mape}


class Training:
    def __init__(self, model_architecture: str, model_name: str):
        # Tạo thư mục cho model_architecture và model_name
        result_architecture = f"{RESULT_DIR}/{model_architecture}"
        if not os.path.exists(result_architecture):
            os.mkdir(result_architecture)

        result_model = f"{result_architecture}/{model_name}"
        if not os.path.exists(result_model):
            os.mkdir(result_model)

        if model_architecture not in config["train"]:
            raise ValueError(f"Unknown model architecture: {model_architecture}")

        self.model_architecture = model_architecture
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module = nn.Module()

        self.config_dict = config["train"][model_architecture][model_name]
        overall_dict = self.config_dict["overall"]
        self.optimizer_dict = self.config_dict["optimizer"]

        self.batch_size = overall_dict["batch_size"]
        self.epochs = overall_dict["epoch"]
        self.dropout = overall_dict["dropout"]

        # The reference STGNN trains on one scalar per node per time step.
        self.input_feature_idx = self.config_dict.get("input_feature_idx", -1)
        self.target_null_value = self.config_dict.get("target_null_value", -1.0)
        self.checkpoint_path = CHECKPOINT_DIR / f"{self.model_architecture.lower()}_{self.model_name}.pt"

    def get_optimizer(self):
        match self.optimizer_dict["name"]:
            case "adam":
                return torch.optim.Adam(
                    self.model.parameters(),
                    **self.optimizer_dict["params"]
                )
            case "sgd":
                return torch.optim.SGD(
                    self.model.parameters(),
                    **self.optimizer_dict["params"]
                )
            case _:
                raise ValueError(f"Unsupported optimizer: {self.model_architecture.lower()}")

    def _prepare_batch(self, X: torch.Tensor, y: torch.Tensor):
        X = X[..., self.input_feature_idx].to(self.device)
        y = y.squeeze(-1).to(self.device)
        return X, y

    def _run_loader(self, loader, optimizer=None):
        is_train = optimizer is not None
        self.model.train(mode=is_train)

        total_loss = 0.0
        total_samples = 0
        preds = []
        targets = []

        for X, y in tqdm(loader, leave=False):
            X, y = self._prepare_batch(X, y)

            if is_train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                y_hat = self.model(X)
                loss = masked_mae(y_hat, y, null_val=self.target_null_value)

                if is_train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    optimizer.step()

            batch_size = X.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            preds.append(y_hat.detach().cpu())
            targets.append(y.detach().cpu())

        avg_loss = total_loss / max(total_samples, 1)
        metrics = regression_metrics(torch.cat(preds, dim=0), torch.cat(targets, dim=0), self.target_null_value)
        return avg_loss, metrics

    def fit(self, train_loader, val_loader):
        optimizer = self.get_optimizer()
        best_val_loss = float("inf")
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            train_loss, train_metrics = self._run_loader(train_loader, optimizer)
            val_loss, val_metrics = self._run_loader(val_loader)

            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"train_mae={train_metrics['mae']:.4f} val_mae={val_metrics['mae']:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "config": self.config_dict,
                        "epoch": epoch,
                        "val_loss": val_loss,
                    },
                    self.checkpoint_path,
                )

        print(f"Best checkpoint saved to {self.checkpoint_path}")

    def test(self, test_loader):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_metrics = self._run_loader(test_loader)
        print(
            f"Test | loss={test_loss:.4f} mae={test_metrics['mae']:.4f} "
            f"rmse={test_metrics['rmse']:.4f} mape={test_metrics['mape']:.4f}"
        )
        return test_loss, test_metrics


class STGNNTraining(Training):
    def __init__(self, model_name: str):
        super().__init__("STGNN", model_name)

        self.K = self.config_dict["K"]
        self.d = self.config_dict["d"]
        self.L = self.config_dict["L"]
        self.F = self.config_dict.get("F", 1)

        if self.F != 1:
            raise ValueError(
                "The reference STGNN implementation expects a single input feature. "
                "Set train.STGNN.<model>.F to 1 or modify src/model/stgnn.py for multi-feature inputs."
            )

        self.model = STGNN(self.F, self.K * self.d, self.L, self.d).to(self.device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="stgnn0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = STGNNTraining(args.model_name)
    train_loader, val_loader, test_loader = get_dataloader(batch_size=trainer.batch_size)
    trainer.fit(train_loader, val_loader)
    trainer.test(test_loader)
