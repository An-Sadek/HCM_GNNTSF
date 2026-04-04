import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from collections.abc import Iterable
from typing import Any, Union

from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# lOAD CONFIG
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

DATA_PATH = config["data"]["raw_path"]
PREPROCESS_PATH = config["data"]["preprocess_path"]


class GenericDataset():
    def __init__(self, file_path: str, preprocess_path: str = PREPROCESS_PATH):
        self.file_path = Path(file_path)
        self.preprocess_path = Path(PREPROCESS_PATH)
        self.name = self.file_path.name
        print(f"\n\nBEGIN === Đang xử lý file {self.name}")

        # Đọc dữ liệu
        df = pd.read_csv(f"{file_path}")

        print("\nShape của file")
        print(df.shape)

        print("\nInfo")
        df.info()

        print("\nMô tả thống kê")
        print(df.describe())

        print("\nCác kiểu dữ liệu")
        print(df.dtypes)

        print("\nSố lượng giá trị bị thiếu")
        print(f"{df.isnull().sum()}")
        print("\nTỉ lệ giá trị bị thiếu")
        print(f"{df.isnull().sum()*100/len(df)}")
        print(f"\nSố lượng giá trị bị thừa: {df.duplicated().sum()}")

        # Lưu vào thuộc tính
        self.df = df

    def start(self):
        pass

    def save(self):
        preprocess_path = str(self.preprocess_path / self.name)
        print(f"\nLưu tại: {preprocess_path}")
        self.df.to_csv(preprocess_path, index=False)

    def end(self):
        del self
        print("\n=== END")

    def toDateTime(self, strformat: str=None):
        pass

    def rename(self, name_dict: dict[str, str]):
        self.df = self.df.rename(columns=name_dict)

    def fillNaN(self, fill_dict: dict[str, Any]):
        print("\nFill dữ liệu thiếu")
        for column, value in fill_dict.items():
            print(f"\tThế giá trị ở cột {column} = {value}")
            self.df[column] = self.df[column].fillna(value)

    def z_scoreStandardization(self, columns: Iterable[str]):
        std_scaler = StandardScaler()
        for column in columns:
            data = self.df[column].values.reshape(-1, 1)
            scaled_data = std_scaler.fit_transform(data).reshape(1, -1)
            self.df[column] = scaled_data

    def ordinalEncoding(self, columns: Iterable[str]):
        print("\nTiến hành Ordinal Encoding")
        encoder = OrdinalEncoder()
        for column in columns:
            data = self.df[column].values.reshape(-1, 1)
            encoded_data = encoder.fit_transform(data).reshape(1, -1)
            self.df[column] = encoded_data

    def oneHotEncoding(self, oh_dict: Union[dict[str, str | None], Iterable[str]]):
        for column, prefix in oh_dict.items():
            if prefix is None:
                prefix = column
            oh_df = pd.get_dummies(df, columns=[column], prefix=prefix)
            oh_df = oh_df.astype(int)
            self.df = pd.concat([self.df, oh_df], axis=1)


class StreetsDataset(GenericDataset):
    def __init__(self, file_path: str):
        GenericDataset.__init__(self, file_path)
        self.start()

    def start(self):
        self.fillNaN({
            "max_velocity": -1,
            "name": "(không tên)"
        })

        # Ordinal encoding bằng data có sẵn
        # 0 là lớn nhất
        self.df["level"] = self.df["level"] - 1
        max_level = self.df["level"].max()
        self.df["level"] = max_level - self.df["level"]

        self.save()
        self.end()


class SegmentsDataset(GenericDataset):
    def __init__(self, file_path: str):
        GenericDataset.__init__(self, file_path)

    def start(self):
        self.fillNaN({
            "max_velocity": -1,
            "street_name": "(không tên)"
        })
        oneHotEncoding(["street_type"])

        self.save()
        self.end()

if __name__ == "__main__":
    file_paths = {
        "streets": f"{DATA_PATH}/streets.csv",
        "segments": f"{DATA_PATH}/segments.csv"
    }
    StreetsDataset(file_paths["streets"])
    SegmentsDataset(file_paths["segments"])

    