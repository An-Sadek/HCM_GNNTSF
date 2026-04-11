import yaml
from pathlib import Path
import pandas as pd
from collections.abc import Iterable
from typing import Any

from sklearn.preprocessing import StandardScaler

# lOAD CONFIG
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

DATA_PATH = config["data"]["path"]["raw"]
PREPROCESS_PATH = config["data"]["path"]["preprocess"]


class GenericDataset():
    def __init__(self, file_path: str, preprocess_path: str = PREPROCESS_PATH):
        self.file_path = Path(file_path)
        self.preprocess_path = Path(preprocess_path)
        self.name = self.file_path.name
        self.config_dict = config["data"]["files"].get(self.name) or {}
        print(self.config_dict)
            
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

        # Chạy start
        self.generic_start()

    def generic_start(self):
        for action, params in self.config_dict.items():
            if params is None:
                continue
            match action:
                case "fillNaN": self.fillNaN(**params)
                case "ordinalEncodingStreetLvl": self.ordinalEncodingStreetLvl(**params)
                case "oneHotEncoding": self.oneHotEncoding(**params)
                case "ordinalEncoding": self.ordinalEncoding(**params)
                case "rename": self.rename(**params)
                case "z_scoreStandardization": self.z_scoreStandardization(**params)
                case "_": continue

    def save(self):
        preprocess_path = str(self.preprocess_path / self.name)
        print(f"\nLưu tại: {preprocess_path}")
        self.df.to_csv(preprocess_path, index=False)

    def end(self):
        self.save()
        del self
        print("\n=== END")

    def check_StaticDynamic(self):
        print("Kiểm tra tính tĩnh động của đồ thị")
        print(f"Thuộc tính\t\t\tTĩnh")

        for column in self.df.columns:
            pass

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
            scaled_data = std_scaler.fit_transform(data)
            self.df[f"{column}_z_score"] = scaled_data.flatten()

    def ordinalEncoding(self, map_dict: dict):
        self.df['LOS_encoded'] = self.df['LOS'].map(map_dict)

    def ordinalEncodingStreetLvl(self, column: str):
        # Ordinal encoding bằng data có sẵn
        # 0 là lớn nhất
        self.df[column] = self.df[column] - 1
        max_level = self.df[column].max()
        self.df[column] = max_level - self.df[column]

    def oneHotEncoding(self, columns: Iterable[str]):
        for column in columns:
            oh_df = pd.get_dummies(self.df[column], prefix=column)
            oh_df = oh_df.astype(int)
            self.df = pd.concat([self.df, oh_df], axis=1)


class StreetsDataset(GenericDataset):
    def __init__(self, file_path: str):
        GenericDataset.__init__(self, file_path)
        self.end()


class SegmentsDataset(GenericDataset):
    def __init__(self, file_path: str):
        GenericDataset.__init__(self, file_path)
        self.end()


class TrainDataset(GenericDataset):
    def __init__(self, file_path: str, segments: SegmentsDataset):
        GenericDataset.__init__(self, file_path)
        self.segments = segments
        self.start()
        self.end()

    def start(self):
        self.weakFilter(**self.config_dict["weakFilter"])
        self.periodExtraction()
        self.mergeZScore(self.segments)

    def weakFilter(self, threshold: int = 10):
        counts = self.df["segment_id"].value_counts()
        valid_segments = counts[counts >= threshold].index
        self.df = self.df[self.df["segment_id"].isin(valid_segments)]
        
        # Thống kê sau khi lọc
        final_counts = self.df["segment_id"].value_counts()
        print(f"\nSố segment sau khi được lọc: {len(final_counts)}")
        print(f"Số record tối thiểu và tối đa của segment: {final_counts.min()}, {final_counts.max()}")
        print(f"Tổng records còn lại: {len(self.df)}")

    def periodExtraction(self):
        period_parts  = self.df['period'].str.split('_', expand=True)
        self.df['hour'] = period_parts [1].astype(int)
        self.df['minute'] = period_parts [2].astype(int)

    def mergeZScore(self, segments: SegmentsDataset):
        #assert length_z_score in segments.df.columns, "Không tìm thấy length_z_score"
        self.df = self.df.merge(
            segments.df[["segment_id", "length_z_score"]],
            on="segment_id",
            how="left"
        )
        

if __name__ == "__main__":
    file_paths = {
        "nodes": f"{DATA_PATH}/nodes.csv",
        "streets": f"{DATA_PATH}/streets.csv",
        "segments": f"{DATA_PATH}/segments.csv",
        "segment_status": f"{DATA_PATH}/segment_status.csv",
        "train": f"{DATA_PATH}/train.csv"
    }
    GenericDataset(file_paths["nodes"])
    GenericDataset(file_paths["segment_status"])
    StreetsDataset(file_paths["streets"])
    segments = SegmentsDataset(file_paths["segments"])
    TrainDataset(file_paths["train"], segments)
