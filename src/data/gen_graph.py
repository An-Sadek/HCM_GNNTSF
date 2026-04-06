from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

PATH_CONFIG = config["data"]["path"]
GRAPH_CONFIG = config["data"]["gen_graph"]

PREPROCESS_PATH = Path(PATH_CONFIG["preprocess"])
DYNAMIC_PATH = Path(PATH_CONFIG["dynamic"])
STATIC_PATH = Path(PATH_CONFIG["static"])
GRAPH_PATH = Path(PATH_CONFIG["graph"])


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    status_df = pd.read_csv(PREPROCESS_PATH / "segment_status.csv")
    segments_df = pd.read_csv(PREPROCESS_PATH / "segments.csv")
    train_df = pd.read_csv(PREPROCESS_PATH / "train.csv")
    return status_df, segments_df, train_df


def parse_datetime_column(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series)


def to_datetime(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = parse_datetime_column(df[column])
    return df


def prepare_dataframes(
    status_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    status_df = to_datetime(status_df.copy(), ["updated_at"])
    segments_df = to_datetime(segments_df.copy(), ["created_at", "updated_at"])
    train_df = to_datetime(train_df.copy(), ["date"])
    return status_df, segments_df, train_df


def build_segment_index(
    segments_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, dict[int, int], dict[int, int]]:
    active_segments = train_df["segment_id"].unique()
    active_segments_df = segments_df[segments_df["segment_id"].isin(active_segments)].copy()
    segment_id_to_node_idx = {
        segment_id: idx for idx, segment_id in enumerate(active_segments)
    }
    node_idx_to_segment_id = {
        idx: segment_id for idx, segment_id in enumerate(active_segments)
    }
    return (
        active_segments_df,
        active_segments,
        segment_id_to_node_idx,
        node_idx_to_segment_id,
    )


def build_edge_index(
    active_segments_df: pd.DataFrame,
    segment_id_to_node_idx: dict[int, int],
) -> torch.Tensor:
    node_to_segments: dict[int, list[int]] = {}

    for _, row in active_segments_df.iterrows():
        segment_idx = segment_id_to_node_idx[row["segment_id"]]
        for node_id in (row["s_node_id"], row["e_node_id"]):
            node_to_segments.setdefault(node_id, []).append(segment_idx)

    num_segments = len(segment_id_to_node_idx)
    edges: set[tuple[int, int]] = set()

    for segment_indices in node_to_segments.values():
        if len(segment_indices) < 2:
            continue
        for i, src in enumerate(segment_indices):
            for dst in segment_indices[i + 1 :]:
                edges.add((src, dst))
                edges.add((dst, src))

    for idx in range(num_segments):
        edges.add((idx, idx))

    edge_list_source = [src for src, _ in edges]
    edge_list_target = [dst for _, dst in edges]
    return torch.tensor([edge_list_source, edge_list_target], dtype=torch.long)


def add_timestamp_features(train_df: pd.DataFrame) -> pd.DataFrame:
    train_df = train_df.copy()
    train_df["timestamp"] = (
        train_df["date"]
        + pd.to_timedelta(train_df["hour"], unit="h")
        + pd.to_timedelta(train_df["minute"], unit="m")
    )
    train_df["weekday"] = train_df["timestamp"].dt.weekday
    return train_df


def create_time_index(train_df: pd.DataFrame, num_segments: int) -> tuple[list[int], pd.DatetimeIndex]:
    all_segments_idx = list(range(num_segments))
    min_time = train_df["timestamp"].min()
    max_time = train_df["timestamp"].max()
    full_time_index = pd.date_range(start=min_time, end=max_time, freq="30min")
    return all_segments_idx, full_time_index


def build_velocity_grid(
    status_df: pd.DataFrame,
    segment_id_to_node_idx: dict[int, int],
    full_time_index: pd.DatetimeIndex,
    all_segments_idx: list[int],
) -> pd.DataFrame:
    status_processed = status_df.copy()
    status_processed["segment_idx"] = status_processed["segment_id"].map(segment_id_to_node_idx)
    status_processed = status_processed.dropna(subset=["segment_idx"])
    status_processed["segment_idx"] = status_processed["segment_idx"].astype(int)
    status_processed = status_processed.set_index("updated_at")

    velocity_grid = (
        status_processed.groupby("segment_idx")["velocity"].resample("30min").mean().unstack("segment_idx")
    )

    if isinstance(velocity_grid.index, pd.DatetimeIndex) and velocity_grid.index.tz is not None:
        velocity_grid.index = velocity_grid.index.tz_convert(None)

    return velocity_grid.reindex(index=full_time_index, columns=all_segments_idx)


def build_dynamic_grid(
    train_df: pd.DataFrame,
    status_df: pd.DataFrame,
    segment_id_to_node_idx: dict[int, int],
    full_time_index: pd.DatetimeIndex,
    num_segments: int,
) -> np.ndarray:
    dynamic_features = ["LOS_encoded", "hour", "minute", "weekday", "velocity"]
    all_segments_idx = list(range(num_segments))

    pivot_base_df = train_df[
        ["timestamp", "segment_id", "LOS_encoded", "hour", "minute", "weekday"]
    ].copy()
    pivot_base_df["segment_idx"] = pivot_base_df["segment_id"].map(segment_id_to_node_idx)
    pivot_base_df = pivot_base_df.dropna(subset=["segment_idx"])
    pivot_base_df["segment_idx"] = pivot_base_df["segment_idx"].astype(int)

    velocity_grid = build_velocity_grid(
        status_df=status_df,
        segment_id_to_node_idx=segment_id_to_node_idx,
        full_time_index=full_time_index,
        all_segments_idx=all_segments_idx,
    )

    data_grid_dynamic = np.zeros(
        (num_segments, len(dynamic_features), len(full_time_index)),
        dtype=np.float32,
    )

    filtered_pivot_base = pivot_base_df[pivot_base_df["timestamp"].isin(full_time_index)]
    filtered_pivot_base = filtered_pivot_base.drop_duplicates(subset=["timestamp", "segment_idx"])

    for feature_idx, feature_name in enumerate(dynamic_features):
        if feature_name == "velocity":
            pivot_feature = velocity_grid
        else:
            pivot_feature = filtered_pivot_base.pivot(
                index="timestamp",
                columns="segment_idx",
                values=feature_name,
            )

        aligned_grid = pivot_feature.reindex(index=full_time_index, columns=all_segments_idx)
        filled_grid = aligned_grid.bfill().ffill().fillna(-1)
        data_grid_dynamic[:, feature_idx, :] = filled_grid.to_numpy(dtype=np.float32).T

    return np.transpose(data_grid_dynamic, (2, 0, 1))


def build_static_grid(
    segments_df: pd.DataFrame,
    segment_id_to_node_idx: dict[int, int],
    num_segments: int,
) -> np.ndarray:
    ohe_street_cols = [
        column for column in segments_df.columns if column.startswith("street_type_")
    ]
    static_features = ["length", "street_level", *ohe_street_cols]

    segments_processed = segments_df.copy()
    segments_processed["segment_idx"] = segments_processed["segment_id"].map(segment_id_to_node_idx)
    segments_processed = segments_processed.dropna(subset=["segment_idx"])
    segments_processed["segment_idx"] = segments_processed["segment_idx"].astype(int)

    sorted_segments = (
        segments_processed.set_index("segment_idx").reindex(list(range(num_segments)))
    )
    data_grid_static = np.zeros((num_segments, len(static_features)), dtype=np.float32)

    for feature_idx, feature_name in tqdm(
        enumerate(static_features),
        total=len(static_features),
        desc="Building static features",
    ):
        if feature_name not in sorted_segments.columns:
            continue

        feature_data = sorted_segments[feature_name].fillna(0).to_numpy()
        if feature_name == "length":
            feature_data = StandardScaler().fit_transform(feature_data.reshape(-1, 1)).squeeze()
        data_grid_static[:, feature_idx] = feature_data.astype(np.float32)

    return data_grid_static


def create_sliding_windows(
    dynamic_grid: np.ndarray,
    past_steps: int,
    future_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_window = past_steps + future_steps
    windows = sliding_window_view(dynamic_grid, window_shape=total_window, axis=0)
    windows = np.transpose(windows, (0, 3, 1, 2)).astype(np.float32)

    x = windows[:, :past_steps, :, :]
    y = windows[:, past_steps:, :, -1:]
    return windows, x, y


def build_graph_tensors(
    edge_index: torch.Tensor,
    static_grid: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_nodes = static_grid.shape[0]
    adj_mtx = torch.zeros((num_nodes, num_nodes), dtype=torch.int64)
    adj_mtx[edge_index[0], edge_index[1]] = 1
    identity = torch.eye(adj_mtx.size(0), dtype=torch.int64)
    a_hat = adj_mtx + identity
    weighted_graph = a_hat.to(torch.float32) @ torch.tensor(static_grid, dtype=torch.float32)
    return adj_mtx, weighted_graph


def save_dynamic_data(
    dynamic_grid: np.ndarray,
    sliding_window: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    with h5py.File(DYNAMIC_PATH, "w") as file:
        file.create_dataset("dynamic", data=dynamic_grid, dtype="float32")
        file.create_dataset("sliding_window", data=sliding_window, dtype="float32")
        file.create_dataset("X", data=x, dtype="float32")
        file.create_dataset("y", data=y, dtype="float32")


def save_static_data(static_grid: np.ndarray) -> None:
    with h5py.File(STATIC_PATH, "w") as file:
        file.create_dataset("static", data=static_grid, dtype="float32")


def save_graph_data(
    edge_index: torch.Tensor,
    adj_mtx: torch.Tensor,
    weighted_graph: torch.Tensor,
) -> None:
    with h5py.File(GRAPH_PATH, "w") as file:
        file.create_dataset("edge_index", data=edge_index.cpu().numpy(), dtype="int64")
        file.create_dataset("adj_mtx", data=adj_mtx.cpu().numpy(), dtype="int64")
        file.create_dataset(
            "weighted_graph",
            data=weighted_graph.cpu().numpy(),
            dtype="float32",
        )


def main() -> None:
    status_df, segments_df, train_df = load_data()
    status_df, segments_df, train_df = prepare_dataframes(status_df, segments_df, train_df)
    train_df = add_timestamp_features(train_df)

    active_segments_df, _, segment_id_to_node_idx, _ = build_segment_index(
        segments_df=segments_df,
        train_df=train_df,
    )
    num_segments = len(segment_id_to_node_idx)

    edge_index = build_edge_index(
        active_segments_df=active_segments_df,
        segment_id_to_node_idx=segment_id_to_node_idx,
    )

    _, full_time_index = create_time_index(train_df=train_df, num_segments=num_segments)

    dynamic_grid = build_dynamic_grid(
        train_df=train_df,
        status_df=status_df,
        segment_id_to_node_idx=segment_id_to_node_idx,
        full_time_index=full_time_index,
        num_segments=num_segments,
    )
    static_grid = build_static_grid(
        segments_df=segments_df,
        segment_id_to_node_idx=segment_id_to_node_idx,
        num_segments=num_segments,
    )
    sliding_window, x, y = create_sliding_windows(
        dynamic_grid=dynamic_grid,
        past_steps=GRAPH_CONFIG["P"],
        future_steps=GRAPH_CONFIG["Q"],
    )
    adj_mtx, weighted_graph = build_graph_tensors(
        edge_index=edge_index,
        static_grid=static_grid,
    )

    save_dynamic_data(dynamic_grid, sliding_window, x, y)
    save_static_data(static_grid)
    save_graph_data(edge_index, adj_mtx, weighted_graph)


if __name__ == "__main__":
    main()
    print("Hoàn thành xây dựng đồ thị")
