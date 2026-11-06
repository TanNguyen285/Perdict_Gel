"""Inference for the trained CNN1D-GRU gel sensor model."""
import json
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.left_padding, 0)))


class GelCNN_GRU(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super().__init__()
        self.cnn = nn.Sequential(
            CausalConv1d(input_size, 32, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            CausalConv1d(32, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.gru = nn.GRU(input_size=64, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head_R = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_L = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, hidden=None):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        features, hidden = self.gru(x, hidden)
        output = torch.cat((self.head_R(features), self.head_L(features)), dim=-1)
        return output, hidden

    def step(self, x, hidden=None):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        features, hidden = self.gru(x[:, -1:, :], hidden)
        output = torch.cat((self.head_R(features), self.head_L(features)), dim=-1)
        return output, hidden


class GelCNNGRUPredictor:
    FEATURES = ["angle_deg", "direction", "dAngle"]
    TARGETS = ["R_kohm", "length_mm"]

    def __init__(self):
        self.model = None
        self.x_scaler = None
        self.y_scaler = None
        self.history = deque()
        self.history_length = 5
        self.hidden = None
        self.previous_angle = None
        self.direction = 1.0
        self.path = None

    @property
    def loaded(self):
        return self.model is not None

    def load(self, path):
        model_path = Path(path)
        config_path = model_path.with_name("config.json")
        x_scaler_path = model_path.with_name("x_scaler.pkl")
        y_scaler_path = model_path.with_name("y_scaler.pkl")

        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        if config.get("features") != self.FEATURES or config.get("targets") != self.TARGETS:
            raise ValueError("Config CNN-GRU không khớp thứ tự features/targets đã huấn luyện.")

        x_scaler = joblib.load(x_scaler_path)
        y_scaler = joblib.load(y_scaler_path)
        if x_scaler.n_features_in_ != len(self.FEATURES) or y_scaler.n_features_in_ != len(self.TARGETS):
            raise ValueError("Scaler CNN-GRU không khớp số lượng features/targets.")

        history_length = int(config["history_length_for_cnn"])
        if history_length < 1:
            raise ValueError("history_length_for_cnn phải lớn hơn 0.")
        model = GelCNN_GRU(input_size=len(self.FEATURES), hidden_size=int(config["hidden_size"]))
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()

        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.history_length = history_length
        self.path = str(model_path)
        self.reset()

    def reset(self):
        self.history.clear()
        self.hidden = None
        self.previous_angle = None
        self.direction = 1.0

    def predict(self, angle):
        if not self.loaded:
            raise RuntimeError("Model CNN-GRU chưa được nạp.")
        angle = float(angle)
        if not np.isfinite(angle):
            raise ValueError("Góc đầu vào CNN-GRU phải là số hữu hạn.")

        delta = 0.0 if self.previous_angle is None else angle - self.previous_angle
        if delta > 1e-6:
            self.direction = 1.0
        elif delta < -1e-6:
            self.direction = -1.0
        self.previous_angle = angle

        self.history.append([angle, self.direction, delta])
        while len(self.history) > self.history_length:
            self.history.popleft()
        scaled = self.x_scaler.transform(np.asarray(self.history, dtype=np.float32))
        x = torch.from_numpy(scaled.astype(np.float32, copy=False)).unsqueeze(0)

        with torch.no_grad():
            output, self.hidden = self.model.step(x, self.hidden)
            self.hidden = self.hidden.detach()
            scaled_target = output[0, -1].cpu().numpy().reshape(1, -1)
        resistance, length = self.y_scaler.inverse_transform(scaled_target)[0]
        return float(length), float(resistance)
