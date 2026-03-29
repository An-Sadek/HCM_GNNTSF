import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import yaml

# lOAD CONFIG
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

SPLIT_RATIO = config["data"]["split_ratio"]

