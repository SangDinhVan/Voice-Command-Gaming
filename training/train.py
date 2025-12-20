import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.datasets.kws_dataset import KWSDataset, LABELS
from training.models.cnn_kws import SmallCNNKWS

def accuracy(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = KWSDataset("data/splits/train.txt", train=True)
    val_ds   = KWSDataset("data/splits/val.txt", train=False)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = SmallCNNKWS(num_classes=len(LABELS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    best_val = 0.0
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(1, 31):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/30 [train]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.item()))

        # val
        model.eval()
        val_acc = 0.0
        n_batches = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/30 [val]"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_acc += accuracy(logits, y)
                n_batches += 1
        val_acc /= max(1, n_batches)

        print(f"Epoch {epoch}: val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            out = ckpt_dir / "kws_cnn_best.pt"
            torch.save({"model": model.state_dict(), "labels": LABELS}, out)
            print("Saved:", out)

    print("Best val acc:", best_val)

if __name__ == "__main__":
    main()
