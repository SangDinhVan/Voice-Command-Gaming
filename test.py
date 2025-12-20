import torchaudio, torch
from pathlib import Path
import random

def rms(path):
    wav, sr = torchaudio.load(str(path))
    if wav.size(0) > 1: wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    return float(torch.sqrt((wav*wav).mean()).item())

root = Path("data/processed")
for cls in ["up","down","left","right"]:
    files = list((root/cls).glob("*.wav"))
    sample = random.sample(files, k=min(200, len(files)))
    vals = [rms(p) for p in sample]
    print(cls, "avg_rms=", sum(vals)/len(vals), "min=", min(vals), "max=", max(vals))
