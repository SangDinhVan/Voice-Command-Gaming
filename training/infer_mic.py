import time
from collections import deque

import numpy as np
import sounddevice as sd
import torch

from training.features.logmel import LogMelExtractor
from training.models.cnn_kws import SmallCNNKWS

LABELS = ["up", "down", "left", "right", "silence", "noise"]
COMMANDS = ["up", "down", "left", "right"]

SR = 16000
WIN_SEC = 1.0
HOP_SEC = 0.15
WIN = int(SR * WIN_SEC)
BLOCK = int(SR * HOP_SEC)

# ====== TUNE HERE ======
# threshold riêng cho từng lệnh (left thấp hơn để dễ bắt hơn)
THRESH_CMD = {
    "up": 0.88,
    "down": 0.88,
    "right": 0.88,
    "left": 0.70,   # <- giảm nếu left khó bắt (0.78~0.85)
}
DEFAULT_CMD_THRESH = 0.88

SMOOTH_K = 3          # keep last 3
NEED = 2              # need >=2/3 agree

COOLDOWN_SEC = 0.45   # tăng nếu vẫn thỉnh thoảng lặp
RESET_FRAMES = 4      # none liên tiếp 4 frame (~0.6s) mới reset last_best
# ======================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ckpt = torch.load("checkpoints/kws_cnn_best.pt", map_location="cpu")
    model = SmallCNNKWS(num_classes=len(LABELS))
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    feat = LogMelExtractor().to(device).eval()

    buffer = np.zeros(WIN, dtype=np.float32)
    history = deque(maxlen=SMOOTH_K)

    last_fire = 0.0
    last_best = "none"
    none_streak = 0

    def callback(indata, frames, t, status):
        nonlocal buffer, last_fire, last_best, none_streak

        x = indata[:, 0].astype(np.float32)
        if x.size == 0:
            return

        # rolling 1s buffer
        buffer = np.roll(buffer, -len(x))
        buffer[-len(x):] = x

        wav = torch.from_numpy(buffer.copy())

        # model inference
        with torch.no_grad():
            logmel = feat(wav.to(device))      # expected (1, 40, T)
            inp = logmel.unsqueeze(0)          # (1, 1, 40, T)
            logits = model(inp)
            prob = torch.softmax(logits, dim=-1)[0]

            # lấy top-2
            top2 = torch.topk(prob, k=2)
            p1, i1 = float(top2.values[0].item()), int(top2.indices[0].item())
            p2, i2 = float(top2.values[1].item()), int(top2.indices[1].item())

            l1, l2 = LABELS[i1], LABELS[i2]

            # rule: nếu right đứng 1 nhưng left sát nút -> chọn left
            if l1 == "right" and l2 == "left" and (p1 - p2) < 0.06:
                pred_label = "left"
                conf = p2
            else:
                pred_label = l1
                conf = p1


        # per-class threshold for commands; other classes treated as none
        if pred_label in COMMANDS:
            thr = THRESH_CMD.get(pred_label, DEFAULT_CMD_THRESH)
            label = pred_label if conf >= thr else "none"
        else:
            label = "none"

        # smoothing history
        history.append(label)

        # vote: pick first command that reaches NEED
        best = "none"
        for cmd in COMMANDS:
            if history.count(cmd) >= NEED:
                best = cmd
                break

        now = time.time()

        # debounce reset: only reset last_best after none for RESET_FRAMES
        if best == "none":
            none_streak += 1
            if none_streak >= RESET_FRAMES:
                last_best = "none"
            return
        else:
            none_streak = 0

        # rising edge trigger
        if best != last_best and (now - last_fire) > COOLDOWN_SEC:
            # conf in print is current frame conf; best is from smoothing
            print(f"{best.upper()} ({conf:.2f})")
            last_fire = now
            history.clear()
            last_best = best

    print("Listening... (Ctrl+C to stop)")
    with sd.InputStream(samplerate=SR, channels=1, blocksize=BLOCK, callback=callback):
        while True:
            sd.sleep(1000)

if __name__ == "__main__":
    main()
