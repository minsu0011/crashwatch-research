from __future__ import annotations

import copy
import gc
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, log_loss


def select_deep_features(
    matrix: np.ndarray,
    y: np.ndarray,
    train_rows: np.ndarray,
    *,
    max_features: int,
) -> np.ndarray:
    """Training-only point-biserial screening for temporal auxiliary models."""
    x = np.asarray(matrix[train_rows], dtype=np.float64)
    target = np.asarray(y[train_rows], dtype=np.float64)
    medians = np.nanmedian(x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    x = np.where(np.isfinite(x), x, medians)
    x -= x.mean(axis=0, keepdims=True)
    target = target - target.mean()
    denominator = np.sqrt(np.sum(x * x, axis=0) * np.sum(target * target))
    correlation = np.divide(np.abs(x.T @ target), denominator, out=np.zeros(x.shape[1]), where=denominator > 1e-12)
    variance = np.var(x, axis=0)
    score = correlation + 1e-8 * np.log1p(variance)
    count = min(max_features, x.shape[1])
    return np.argsort(score)[-count:][::-1].astype(np.int32)


def build_window_indices(tickers: np.ndarray, dates: np.ndarray, sequence_length: int) -> np.ndarray:
    """Return row-id windows ending at each row; -1 marks insufficient history."""
    tickers = np.asarray(tickers).astype(str)
    dates = np.asarray(dates)
    windows = np.full((len(tickers), sequence_length), -1, dtype=np.int32)
    order = np.lexsort((dates, tickers))
    sorted_tickers = tickers[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_tickers[end] == sorted_tickers[start]:
            end += 1
        rows = order[start:end]
        for position in range(sequence_length - 1, len(rows)):
            current = rows[position]
            windows[current] = rows[position - sequence_length + 1 : position + 1]
        start = end
    return windows


@dataclass
class DeepPreprocessor:
    selected_columns: np.ndarray
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray, train_rows: np.ndarray, selected_columns: np.ndarray) -> "DeepPreprocessor":
        values = np.asarray(matrix[train_rows][:, selected_columns], dtype=np.float64)
        medians = np.nanmedian(values, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        values = np.where(np.isfinite(values), values, medians)
        means = values.mean(axis=0)
        scales = values.std(axis=0)
        scales = np.where(np.isfinite(scales) & (scales > 1e-6), scales, 1.0)
        return cls(selected_columns.astype(np.int32), medians.astype(np.float32), means.astype(np.float32), scales.astype(np.float32))

    def transform_windows(self, matrix: np.ndarray, window_rows: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix[window_rows][..., self.selected_columns], dtype=np.float32)
        missing = ~np.isfinite(values)
        values = np.where(missing, self.medians, values)
        values = (values - self.means) / self.scales
        missing_ratio = missing.mean(axis=2, keepdims=True).astype(np.float32)
        return np.concatenate([values.astype(np.float32), missing_ratio], axis=2)

    def payload(self) -> dict[str, Any]:
        return {
            "selected_columns": self.selected_columns.tolist(),
            "medians": self.medians.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
        }


@dataclass
class DeepFitResult:
    family: str
    state_dict: dict[str, Any]
    config: dict[str, Any]
    batch_size: int
    epochs: int
    best_validation_logloss: float
    selected_columns: np.ndarray
    preprocessor: DeepPreprocessor
    actual_backend: str

    def save(self, path: Path) -> None:
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "family": self.family,
                "state_dict": self.state_dict,
                "config": self.config,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "selected_columns": self.selected_columns.tolist(),
                "preprocessor": self.preprocessor.payload(),
                "backend": self.actual_backend,
            },
            path,
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(family: str, input_dim: int, sequence_length: int, config: dict[str, Any]):
    import torch
    from torch import nn

    class ResidualTemporalBlock(nn.Module):
        def __init__(self, channels: int, dilation: int, dropout: float) -> None:
            super().__init__()
            padding = dilation
            self.network = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
                nn.GELU(),
                nn.BatchNorm1d(channels),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
                nn.GELU(),
                nn.BatchNorm1d(channels),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            output = self.network(x)
            if output.shape[-1] != x.shape[-1]:
                output = output[..., : x.shape[-1]]
            return x + output

    class TemporalCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = int(config.get("channels", 192))
            dropout = float(config.get("dropout", 0.15))
            self.input = nn.Conv1d(input_dim, channels, kernel_size=1)
            self.blocks = nn.Sequential(*[ResidualTemporalBlock(channels, dilation, dropout) for dilation in (1, 2, 4, 8)])
            self.attention = nn.Sequential(nn.Conv1d(channels, 1, kernel_size=1), nn.Softmax(dim=-1))
            self.head = nn.Sequential(
                nn.Linear(channels, channels // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(channels // 2, 1)
            )

        def forward(self, x):
            x = self.input(x.transpose(1, 2))
            x = self.blocks(x)
            weights = self.attention(x)
            pooled = torch.sum(x * weights, dim=-1)
            return self.head(pooled).squeeze(-1)

    class TemporalTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            d_model = int(config.get("d_model", 192))
            nhead = int(config.get("nhead", 8))
            layers = int(config.get("layers", 3))
            dropout = float(config.get("dropout", 0.15))
            self.input = nn.Linear(input_dim, d_model)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            self.position = nn.Parameter(torch.zeros(1, sequence_length + 1, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))
            nn.init.normal_(self.position, std=0.02)
            nn.init.normal_(self.cls, std=0.02)

        def forward(self, x):
            x = self.input(x)
            cls = self.cls.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1) + self.position[:, : x.shape[1] + 1]
            x = self.encoder(x)
            return self.head(self.norm(x[:, 0])).squeeze(-1)

    if family == "tcn":
        return TemporalCNN()
    if family == "transformer":
        return TemporalTransformer()
    raise ValueError(family)


def _valid_rows(rows: np.ndarray, windows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int32)
    return rows[np.all(windows[rows] >= 0, axis=1)]


def _auto_batch_size(model, input_dim: int, sequence_length: int, device, use_bf16: bool, memory_fraction: float) -> int:
    import torch

    if device.type != "cuda":
        return 128
    torch.cuda.set_per_process_memory_fraction(float(memory_fraction), device=device)
    total = torch.cuda.get_device_properties(device).total_memory
    candidates = [4096, 3072, 2048, 1536, 1024, 768, 512, 384, 256]
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    was_training = model.training
    model.eval()
    for batch_size in candidates:
        try:
            torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)
            sample = torch.randn(batch_size, sequence_length, input_dim, device=device)
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss = model(sample).float().square().mean()
            loss.backward()
            allocated = torch.cuda.max_memory_allocated(device)
            del sample, loss
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if allocated / total <= memory_fraction:
                model.train(was_training)
                return batch_size
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
    model.train(was_training)
    return 128


def fit_deep_model(
    family: str,
    matrix: np.ndarray,
    y: np.ndarray,
    windows: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    *,
    seed: int,
    sequence_length: int,
    max_features: int,
    max_epochs: int,
    patience: int,
    threads: int,
    memory_fraction: float,
    config: dict[str, Any] | None = None,
) -> DeepFitResult:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    config = dict(config or {})
    _seed_everything(seed)
    torch.set_num_threads(max(1, threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("deep canonical tasks require CUDA; PUBG mode leaves them pending")
    selected = select_deep_features(matrix, y, train_rows, max_features=max_features)
    preprocessor = DeepPreprocessor.fit(matrix, train_rows, selected)
    train_rows = _valid_rows(train_rows, windows)
    validation_rows = _valid_rows(validation_rows, windows)
    if len(train_rows) < 1000 or len(validation_rows) < 100:
        raise RuntimeError("insufficient sequence rows")

    class SequenceDataset(Dataset):
        def __init__(self, rows: np.ndarray) -> None:
            self.rows = np.asarray(rows, dtype=np.int32)

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = int(self.rows[index])
            window = windows[row : row + 1]
            values = preprocessor.transform_windows(matrix, window)[0]
            return torch.from_numpy(values), torch.tensor(float(y[row]), dtype=torch.float32), row

    model_config = {
        "channels": 192,
        "d_model": 192,
        "nhead": 8,
        "layers": 3,
        "dropout": 0.15,
        "learning_rate": 8e-4 if family == "tcn" else 5e-4,
        "weight_decay": 1e-4,
        **config,
    }
    input_dim = len(selected) + 1
    model = _build_model(family, input_dim, sequence_length, model_config).to(device)
    use_bf16 = bool(torch.cuda.is_bf16_supported())
    batch_size = _auto_batch_size(model, input_dim, sequence_length, device, use_bf16, memory_fraction)
    workers = 0  # Windows spawn would copy the large feature matrix.
    train_loader = DataLoader(SequenceDataset(train_rows), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    validation_loader = DataLoader(SequenceDataset(validation_rows), batch_size=batch_size * 2, shuffle=False, num_workers=workers, pin_memory=True)

    positive = float(np.sum(y[train_rows] == 1))
    negative = float(np.sum(y[train_rows] == 0))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / max(1.0, positive), device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    best_state = copy.deepcopy(model.state_dict())
    best_score = math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for features, target, _ in train_loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(features)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        model.eval()
        validation_target: list[np.ndarray] = []
        validation_prediction: list[np.ndarray] = []
        with torch.inference_mode():
            for features, target, _ in validation_loader:
                features = features.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    probability = torch.sigmoid(model(features)).float().cpu().numpy()
                validation_target.append(target.numpy())
                validation_prediction.append(probability)
        target_np = np.concatenate(validation_target)
        probability_np = np.clip(np.concatenate(validation_prediction), 1e-7, 1 - 1e-7)
        score = float(log_loss(target_np, probability_np, labels=[0, 1]))
        # Tie-break toward better ranking when probability loss is nearly equal.
        pr_auc = float(average_precision_score(target_np, probability_np)) if len(np.unique(target_np)) == 2 else 0.0
        selection = score - 1e-3 * pr_auc
        if selection < best_score - 1e-5:
            best_score = selection
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    result = DeepFitResult(
        family=family,
        state_dict={key: value.detach().cpu() for key, value in best_state.items()},
        config=model_config,
        batch_size=batch_size,
        epochs=best_epoch,
        best_validation_logloss=best_score,
        selected_columns=selected,
        preprocessor=preprocessor,
        actual_backend="cuda_bf16" if use_bf16 else "cuda_fp16",
    )
    del model, train_loader, validation_loader
    gc.collect()
    torch.cuda.empty_cache()
    return result


def predict_deep_model(
    fitted: DeepFitResult,
    matrix: np.ndarray,
    windows: np.ndarray,
    rows: np.ndarray,
    *,
    sequence_length: int,
    threads: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader, Dataset

    torch.set_num_threads(max(1, threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required for deep prediction")
    valid_rows = _valid_rows(rows, windows)
    input_dim = len(fitted.selected_columns) + 1
    model = _build_model(fitted.family, input_dim, sequence_length, fitted.config).to(device)
    model.load_state_dict(fitted.state_dict)
    model.eval()
    amp_dtype = torch.bfloat16 if "bf16" in fitted.actual_backend else torch.float16

    class PredictDataset(Dataset):
        def __len__(self):
            return len(valid_rows)

        def __getitem__(self, index: int):
            row = int(valid_rows[index])
            values = fitted.preprocessor.transform_windows(matrix, windows[row : row + 1])[0]
            return torch.from_numpy(values), row

    loader = DataLoader(PredictDataset(), batch_size=fitted.batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)
    predictions: list[np.ndarray] = []
    output_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for features, batch_rows in loader:
            features = features.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                probability = torch.sigmoid(model(features)).float().cpu().numpy()
            predictions.append(probability)
            output_rows.append(batch_rows.numpy())
    del model, loader
    gc.collect()
    torch.cuda.empty_cache()
    return np.concatenate(output_rows).astype(np.int32), np.concatenate(predictions).astype(np.float32)
