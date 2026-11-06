
import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. CONFIG
# =========================================================

CSV_PATH = r"C:\Users\ThisPC\Desktop\testapp\CNN1D\gel_clean.csv"
SAVE_DIR = "gel_cnn1d_gru"

EPOCHS = 300
LR = 1e-3
HIDDEN_SIZE = 64
TRAIN_RATIO = 0.8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)


# =========================================================
# 2. LOAD DATA
# =========================================================

df = pd.read_csv(CSV_PATH)

required_cols = [
    "cycle_id", "step", "direction",
    "angle_deg", "length_mm", "R_kohm"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"CSV thiếu các cột: {missing}")

df = df[required_cols].copy()

for col in ["step", "direction", "angle_deg", "length_mm", "R_kohm"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna().copy()
df = df[
    df["R_kohm"].between(5, 20)
].copy()

# Giữ thứ tự chu kỳ xuất hiện trong file.
cycle_ids = df["cycle_id"].drop_duplicates().tolist()

if len(cycle_ids) < 2:
    raise ValueError(
        "Cần ít nhất 2 chu kỳ để chia train/validation. "
        "Một chu kỳ chưa đủ để đánh giá khả năng tổng quát hóa."
    )

# Sắp xếp step bên trong từng chu kỳ.
groups = []
for cid in cycle_ids:
    g = df[df["cycle_id"] == cid].sort_values("step").copy()

    # Biến thiên góc trong cùng chu kỳ.
    g["dAngle"] = g["angle_deg"].diff().fillna(0)

    groups.append(g)


# =========================================================
# 3. CHRONOLOGICAL TRAIN / VALIDATION SPLIT
# =========================================================

split_idx = int(len(groups) * TRAIN_RATIO)
split_idx = min(max(split_idx, 1), len(groups) - 1)

train_groups = groups[:split_idx]
val_groups = groups[split_idx:]

train_df = pd.concat(train_groups, ignore_index=True)
val_df = pd.concat(val_groups, ignore_index=True)

print("Train cycles:", [g["cycle_id"].iloc[0] for g in train_groups])
print("Val cycles:", [g["cycle_id"].iloc[0] for g in val_groups])


# =========================================================
# 4. INPUT / TARGET
# =========================================================

# Không đưa R_kohm vào input vì cần dự đoán R_kohm.
FEATURES = [
    "angle_deg",
    "direction",
    "dAngle"
]

TARGETS = [
    "R_kohm",
    "length_mm"
]

X_train_raw = train_df[FEATURES].values.astype(np.float32)
Y_train_raw = train_df[TARGETS].values.astype(np.float32)

X_val_raw = val_df[FEATURES].values.astype(np.float32)
Y_val_raw = val_df[TARGETS].values.astype(np.float32)


# =========================================================
# 5. NORMALIZATION
# Chỉ fit scaler trên TRAIN để tránh data leakage.
# =========================================================

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train_raw)
Y_train = y_scaler.fit_transform(Y_train_raw)

X_val = x_scaler.transform(X_val_raw)
Y_val = y_scaler.transform(Y_val_raw)


# =========================================================
# 6. CNN 1D + GRU MODEL
# =========================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.left_padding = kernel_size - 1

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0
        )

    def forward(self, x):
        x = nn.functional.pad(
            x,
            (self.left_padding, 0)
        )
        return self.conv(x)


class GelCNN_GRU(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super().__init__()

        self.cnn = nn.Sequential(
            CausalConv1d(input_size, 32, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            CausalConv1d(32, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Head 1: dự đoán điện trở.
        self.head_R = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Head 2: dự đoán chiều dài.
        self.head_L = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, hidden=None):
        # x: [batch, time, features]

        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        features, hidden = self.gru(x, hidden)

        pred_R = self.head_R(features)
        pred_L = self.head_L(features)

        output = torch.cat(
            [pred_R, pred_L],
            dim=-1
        )

        return output, hidden


model = GelCNN_GRU(
    input_size=len(FEATURES),
    hidden_size=HIDDEN_SIZE
).to(DEVICE)


# =========================================================
# 7. TENSOR
# Một stream liên tục, GRU giữ hidden state qua các chu kỳ.
# =========================================================

X_train_t = torch.tensor(
    X_train, dtype=torch.float32
).unsqueeze(0).to(DEVICE)

Y_train_t = torch.tensor(
    Y_train, dtype=torch.float32
).unsqueeze(0).to(DEVICE)

X_val_t = torch.tensor(
    X_val, dtype=torch.float32
).unsqueeze(0).to(DEVICE)

Y_val_t = torch.tensor(
    Y_val, dtype=torch.float32
).unsqueeze(0).to(DEVICE)


# =========================================================
# 8. TRAINING
# =========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-5
)

criterion = nn.MSELoss()

best_val_loss = float("inf")
best_model_path = os.path.join(SAVE_DIR, "best_model.pth")

for epoch in range(EPOCHS):

    model.train()

    optimizer.zero_grad()

    pred_train, _ = model(X_train_t)

    loss_train = criterion(
        pred_train,
        Y_train_t
    )

    loss_train.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    # -------------------------------------
    # Validation
    # -------------------------------------

    model.eval()

    with torch.no_grad():

        # Tạo trạng thái nhớ sau khi chạy hết train stream.
        # Validation tiếp tục từ trạng thái này.
        _, train_hidden = model(X_train_t)

        pred_val, _ = model(
            X_val_t,
            train_hidden.detach()
        )

        loss_val = criterion(
            pred_val,
            Y_val_t
        ).item()

    if loss_val < best_val_loss:
        best_val_loss = loss_val

        torch.save(
            model.state_dict(),
            best_model_path
        )

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {loss_train.item():.6f} "
            f"Val Loss: {loss_val:.6f}"
        )


# =========================================================
# 9. EVALUATION
# =========================================================

model.load_state_dict(
    torch.load(best_model_path, map_location=DEVICE)
)

model.eval()

with torch.no_grad():

    _, train_hidden = model(X_train_t)

    pred_val, _ = model(
        X_val_t,
        train_hidden.detach()
    )

    pred_val = pred_val.squeeze(0).cpu().numpy()

pred_val_real = y_scaler.inverse_transform(pred_val)

y_true = Y_val_raw

pred_R = pred_val_real[:, 0]
pred_L = pred_val_real[:, 1]

true_R = y_true[:, 0]
true_L = y_true[:, 1]


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")


evaluate("Resistance R (kOhm)", true_R, pred_R)
evaluate("Length L (mm)", true_L, pred_L)


# =========================================================
# 10. SAVE MODEL + SCALERS + CONFIG
# =========================================================

joblib.dump(
    x_scaler,
    os.path.join(SAVE_DIR, "x_scaler.pkl")
)

joblib.dump(
    y_scaler,
    os.path.join(SAVE_DIR, "y_scaler.pkl")
)

config = {
    "features": FEATURES,
    "targets": TARGETS,
    "hidden_size": HIDDEN_SIZE,
    "history_length_for_cnn": 5
}

with open(
    os.path.join(SAVE_DIR, "config.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(config, f, indent=4)

print("\nTraining completed.")
print("Saved to:", SAVE_DIR)
