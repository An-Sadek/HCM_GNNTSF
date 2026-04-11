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
        """
        Bắt đầu đọc từ từ điển config và thực thi theo thứ tự
        """
        for action, params in self.config_dict.items():
            if params is None:
                continue
            match action:
                case "fillNaN": self.fillNaN(**params)
                case "ordinalEncodingStreetLvl": self.ordinalEncodingStreetLvl(**params)
                case "oneHotEncoding": self.oneHotEncoding(**params)
                case "ordinalEncoding": self.ordinalEncoding(**params)
                case "rename": self.rename(**params)
                case "_": continue

        self.end()

    def save(self):
        """
        Lưu lại các file đã xử lý
        """
        self.preprocess_path.mkdir(parents=True, exist_ok=True)
        preprocess_path = str(self.preprocess_path / self.name)
        print(f"\nLưu tại: {preprocess_path}")
        self.df.to_csv(preprocess_path, index=False)

    def end(self):
        """
        Hàm hủy
        """
        self.save()
        del self
        print("\n=== END")

    def rename(self, name_dict: dict[str, str]):
        """
        Đổi tên cột

        Args:
            name_dict: Từ điển k, v là tên cột muốn đổi và tên mới
        """
        self.df = self.df.rename(columns=name_dict)

    def fillNaN(self, fill_dict: dict[str, Any]):
        """
        Thế giá trị bị thiếu

        Args:
            fill_dict: Từ điển gồm từ khóa là cột bị thiếu và giá trị muốn thế
        """
        print("\nFill dữ liệu thiếu")
        for column, value in fill_dict.items():
            print(f"\tThế giá trị ở cột {column} = {value}")
            self.df[column] = self.df[column].fillna(value)

    def ordinalEncoding(self, map_dict: dict):
        """
        Ordinal Encoding sử dụng dictionary

        Args:
            map_dict: Từ điển thay thế giá trị
        """
        self.df['LOS_encoded'] = self.df['LOS'].map(map_dict)

    def ordinalEncodingStreetLvl(self, column: str="street_level"):
        """
        Oridnal Encoding thuộc tính street level cho segments.csv và train.csv

        Args:
            column: Tên cột street đó, trong trường hợp bị sai
        """
        # Ordinal encoding bằng data có sẵn
        # 0 là lớn nhất
        self.df[column] = self.df[column] - 1
        max_level = self.df[column].max()
        self.df[column] = max_level - self.df[column]

    def oneHotEncoding(self, columns: Iterable[str]):
        """
        One-hot encoding các thuộc tính

        Args:
            columns: Danh sách các cột muốn one-hot
        """
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

    def weakFilter(self, threshold: int = 10):
        """
        Lọc các segment có số lượng record ít

        Args:
            threshold: Ngưỡng đạt chuẩn, giữ các segments nếu >= ngưỡng này
        """
        counts = self.df["segment_id"].value_counts()
        valid_segments = counts[counts >= threshold].index
        self.df = self.df[self.df["segment_id"].isin(valid_segments)]
        
        # Thống kê sau khi lọc
        final_counts = self.df["segment_id"].value_counts()
        print(f"\nSố segment sau khi được lọc: {len(final_counts)}")
        print(f"Số record tối thiểu và tối đa của segment: {final_counts.min()}, {final_counts.max()}")
        print(f"Tổng records còn lại: {len(self.df)}")

    def periodExtraction(self):
        """
        Trích xuất thuộc tính "period" trong dataset
        """
        period_parts  = self.df['period'].str.split('_', expand=True)
        self.df['hour'] = period_parts [1].astype(int)
        self.df['minute'] = period_parts [2].astype(int)
        

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
