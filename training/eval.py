import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from training.datasets.kws_dataset import KWSDataset
from training.models.cnn_kws import SmallCNNKWS

LABELS = ["up", "down", "left", "right", "silence", "noise"]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ===== Load model =====
    ckpt = torch.load("checkpoints/kws_cnn_best.pt", map_location="cpu")
    model = SmallCNNKWS(num_classes=len(LABELS))
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    # ===== Dataset =====
    test_ds = KWSDataset("data/splits/test.txt",train=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    # ===== Eval =====
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # ===== Accuracy =====
    acc = (all_preds == all_targets).mean()
    print(f"\nOverall Accuracy: {acc * 100:.2f}%\n")

    # ===== Confusion Matrix =====
    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    # Pretty print per-class
    print("\nPer-class report:")
    print(classification_report(
        all_targets,
        all_preds,
        target_names=LABELS,
        digits=4
    ))

    # ===== Focus on LEFT =====
    left_id = LABELS.index("left")
    left_total = (all_targets == left_id).sum()
    left_correct = ((all_targets == left_id) & (all_preds == left_id)).sum()

    print(f"\nLEFT stats:")
    print(f"  Total LEFT samples : {left_total}")
    print(f"  Correct LEFT       : {left_correct}")
    print(f"  LEFT Accuracy      : {100 * left_correct / max(left_total,1):.2f}%")

    print("\nLEFT misclassified as:")
    for i, lbl in enumerate(LABELS):
        if i != left_id:
            cnt = ((all_targets == left_id) & (all_preds == i)).sum()
            if cnt > 0:
                print(f"  -> {lbl}: {cnt}")

if __name__ == "__main__":
    main()
