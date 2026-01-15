import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from utils import ensure_dirs, load_gesture_labels

IN_DIM = 64

class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def load_all_npz():
    files = sorted(glob.glob(os.path.join("data", "raw", "*.npz")))
    if not files:
        raise RuntimeError("No .npz files found in data/raw/. Run collect_data.py first.")
    Xs, ys = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        Xs.append(d["X"].astype(np.float32))
        ys.append(d["y"].astype(np.int64))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y, files

def accuracy(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

def main():
    ensure_dirs()
    gestures = load_gesture_labels()
    if not gestures:
        raise RuntimeError("Missing data/gesture_labels.json. Run collect_data.py once.")

    num_classes = len(gestures)
    X, y, files = load_all_npz()

    if X.shape[1] != IN_DIM:
        raise RuntimeError(f"Feature dim mismatch: expected {IN_DIM}, got {X.shape[1]}")

    print(f"Loaded {len(y)} samples from {len(files)} files.")
    print("Class counts:")
    for i, g in enumerate(gestures):
        print(f"  {i}: {g} -> {(y==i).sum()}")

    ds = GestureDataset(X, y)
    n_total = len(ds)
    n_val = max(1, int(0.15 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=IN_DIM, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = 0.0
    epochs = 35

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy(logits.detach(), yb)

        train_loss /= max(1, len(train_loader))
        train_acc /= max(1, len(train_loader))

        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_acc += accuracy(logits, yb)
        val_acc /= max(1, len(val_loader))

        print(f"Epoch {epoch:02d}/{epochs} | loss {train_loss:.4f} | train_acc {train_acc:.3f} | val_acc {val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            save_path = os.path.join("models", "model.pth")
            torch.save({
                "model_state": model.state_dict(),
                "gestures": gestures,
                "in_dim": IN_DIM,
                "num_classes": num_classes,
            }, save_path)
            print(f"  Saved best model -> {save_path} (val_acc={best_val:.3f})")

    print(f"Done. Best val_acc={best_val:.3f}")

if __name__ == "__main__":
    main()
