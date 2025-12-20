import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from training.datasets.audio_utils import (
    load_wav_16k_1s, time_shift, random_gain, mix_background
)
from training.features.logmel import LogMelExtractor

LABELS = ["up", "down", "left", "right", "silence", "noise"]

class KWSDataset(Dataset):
    def __init__(
        self,
        split_txt: str,
        root: str = "data",
        train: bool = True,
        bg_noise_dir: str = "data/background_noise",
        p_mix_bg: float = 0.6,
    ):
        self.root = Path(root)
        self.train = train
        self.p_mix_bg = p_mix_bg

        # read split
        items: List[Tuple[str, int]] = []
        with open(split_txt, "r", encoding="utf-8") as f:
            for line in f:
                rel, y = line.strip().split()
                items.append((rel, int(y)))
        self.items = items

        self.bg_files = []
        bgp = Path(bg_noise_dir)
        if bgp.exists():
            self.bg_files = sorted(list(bgp.glob("*.wav")))

        self.feat = LogMelExtractor()

    def __len__(self):
        return len(self.items)

    def _sample_bg(self):
        if not self.bg_files:
            return None
        return str(random.choice(self.bg_files))

    def __getitem__(self, idx: int):
        rel, y = self.items[idx]
        path = self.root / rel
        wav = load_wav_16k_1s(str(path))

        if self.train:
            wav = time_shift(wav, 100)
            wav = random_gain(wav, 0.8, 1.2)

            if self.bg_files and random.random() < self.p_mix_bg:
                bgp = self._sample_bg()
                bg_wav = load_wav_16k_1s(bgp)
                wav = mix_background(wav, bg_wav)

        x = self.feat(wav)

        if x.dim() == 2:
            x = x.unsqueeze(0)       

        if x.dim() == 4:
            x = x.squeeze(0)         

        return x.float(), torch.tensor(y, dtype=torch.long)

