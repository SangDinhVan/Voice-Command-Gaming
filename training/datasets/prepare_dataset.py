import os
import random
import shutil
from pathlib import Path

import torch
import torchaudio

from training.datasets.audio_utils import SR, TARGET_LEN, load_wav_16k_1s, random_crop_1s_from_long

RANDOM_SEED = 1337
random.seed(RANDOM_SEED)

LABELS = ["up", "down", "left", "right", "silence", "noise"]
LABEL2ID = {k: i for i, k in enumerate(LABELS)}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_wavs(folder: Path):
    return sorted([p for p in folder.rglob("*.wav") if p.is_file()])

def write_split(file_path: Path, items):
    with file_path.open("w", encoding="utf-8") as f:
        for rel, y in items:
            f.write(f"{rel} {y}\n")

def main():
    raw_root = Path("data/raw")
    proc_root = Path("data/processed")
    splits_root = Path("data/splits")
    bg_out_root = Path("data/_background_noise_")

    if not raw_root.exists():
        raise FileNotFoundError(f"Không thấy {raw_root}. Bạn hãy giải nén Speech Commands vào đó.")

    ensure_dir(proc_root)
    ensure_dir(splits_root)
    ensure_dir(bg_out_root)
    for lbl in LABELS:
        ensure_dir(proc_root / lbl)

    # 1) copy 4 command folders -> processed
    commands = ["up", "down", "left", "right"]
    all_items = []  # (processed_rel_path, label_id)

    print("Copy commands...")
    for cmd in commands:
        src = raw_root / cmd
        if not src.exists():
            raise FileNotFoundError(f"Thiếu folder {src}")
        files = list_wavs(src)
        for p in files:
            dst = proc_root / cmd / p.name
            if not dst.exists():
                shutil.copy2(p, dst)
            rel = dst.relative_to(Path("data"))
            all_items.append((str(rel).replace("\\", "/"), LABEL2ID[cmd]))

    # 2) build 1s background noise clips from _background_noise_
    bg_src = raw_root / "_background_noise_"
    bg_clips = []
    if bg_src.exists():
        print("Build 1s background clips...")
        bg_files = list_wavs(bg_src)
        clip_idx = 0
        for bf in bg_files:
            wav, sr = torchaudio.load(str(bf))
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != SR:
                wav = torchaudio.functional.resample(wav, sr, SR)
            wav = wav.squeeze(0)

            # cut multiple 1s random clips
            for _ in range(30):
                clip = random_crop_1s_from_long(wav)
                # reduce volume to resemble "silence"
                clip = clip * random.uniform(0.02, 0.08)
                out = bg_out_root / f"bg_{clip_idx:06d}.wav"
                torchaudio.save(str(out), clip.unsqueeze(0), SR)
                bg_clips.append(out)
                clip_idx += 1
    else:
        print("Warning: không có _background_noise_. Vẫn chạy được nhưng silence/noise sẽ kém hơn.")

    # 3) create silence class (from background clips)
    print("Create silence samples...")
    silence_n = min(2000, len(bg_clips)) if bg_clips else 0
    for i in range(silence_n):
        src = bg_clips[i]
        dst = proc_root / "silence" / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        rel = dst.relative_to(Path("data"))
        all_items.append((str(rel).replace("\\", "/"), LABEL2ID["silence"]))

    # 4) create noise class (unknown words + (optional) extra bg clips at higher volume)
    print("Create noise samples...")
    cmd_set = set(commands) | {"_background_noise_"}
    unknown_folders = [p for p in raw_root.iterdir() if p.is_dir() and p.name not in cmd_set]
    unknown_files = []
    for uf in unknown_folders:
        unknown_files += list_wavs(uf)
    random.shuffle(unknown_files)

    noise_target = max(3000, len([x for x in all_items if x[1] == LABEL2ID["up"]]))  # rough target
    noise_count = 0
    for p in unknown_files[:noise_target]:
        dst = proc_root / "noise" / f"{p.parent.name}_{p.name}"
        if not dst.exists():
            # normalize to 1s 16k
            wav = load_wav_16k_1s(str(p))
            torchaudio.save(str(dst), wav.unsqueeze(0), SR)
        rel = dst.relative_to(Path("data"))
        all_items.append((str(rel).replace("\\", "/"), LABEL2ID["noise"]))
        noise_count += 1

    # add extra background as "noise" (louder than silence)
    if bg_clips:
        extra = min(1500, len(bg_clips))
        for i in range(extra):
            wav, _ = torchaudio.load(str(bg_clips[-1 - i]))
            wav = wav.squeeze(0) * random.uniform(0.2, 0.6)  # louder
            dst = proc_root / "noise" / f"bgnoise_{i:06d}.wav"
            torchaudio.save(str(dst), wav.unsqueeze(0), SR)
            rel = dst.relative_to(Path("data"))
            all_items.append((str(rel).replace("\\", "/"), LABEL2ID["noise"]))

    print(f"Total samples: {len(all_items)}")

    # 5) split train/val/test
    random.shuffle(all_items)
    n = len(all_items)
    n_train = int(0.80 * n)
    n_val = int(0.10 * n)

    train = all_items[:n_train]
    val = all_items[n_train:n_train + n_val]
    test = all_items[n_train + n_val:]

    write_split(splits_root / "train.txt", train)
    write_split(splits_root / "val.txt", val)
    write_split(splits_root / "test.txt", test)

    print("Saved splits:")
    print(" -", splits_root / "train.txt")
    print(" -", splits_root / "val.txt")
    print(" -", splits_root / "test.txt")
    print("Done.")

if __name__ == "__main__":
    main()

# python -m training.datasets.prepare_dataset