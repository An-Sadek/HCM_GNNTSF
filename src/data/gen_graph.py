import yaml
import h5py
from tqdm import tqdm
from pathlib import Path
from typing import Union
from collections.abc import Iterable

from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import pandas as pd

from torch_geometric.utils import to_dense_adj
import torch
from sklearn.preprocessing import StandardScaler

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

PATH_CONFIG = config["data"]["path"]
PREPROCESS_PATH = PATH_CONFIG["preprocess"]

def load_data():
    status_df = pd.read_csv(f"{PREPROCESS_PATH}/segment_status.csv")
    segments_df = pd.read_csv(f"{PREPROCESS_PATH}/segments.csv")
    train_df = pd.read_csv(f"{PREPROCESS_PATH}/train.csv")
    return status_df, segments_df, train_df

def to_datetime(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = pd.to_datetime(df[column])
    return df