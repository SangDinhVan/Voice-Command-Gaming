import random
import torch
import torchaudio

SR = 16000
TARGET_LEN = SR  # 1 second

def load_wav_16k_1s(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)  # wav: (C,T)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)

    wav = wav.squeeze(0)  # (T,)

    if wav.numel() < TARGET_LEN:
        pad = TARGET_LEN - wav.numel()
        wav = torch.nn.functional.pad(wav, (0, pad))
    elif wav.numel() > TARGET_LEN:
        start = (wav.numel() - TARGET_LEN) // 2
        wav = wav[start:start + TARGET_LEN]

    return wav

def random_crop_1s_from_long(wav: torch.Tensor) -> torch.Tensor:
    if wav.numel() < TARGET_LEN:
        pad = TARGET_LEN - wav.numel()
        wav = torch.nn.functional.pad(wav, (0, pad))
        return wav
    if wav.numel() == TARGET_LEN:
        return wav
    start = random.randint(0, wav.numel() - TARGET_LEN)
    return wav[start:start + TARGET_LEN]

def time_shift(wav: torch.Tensor, shift_ms: int = 100) -> torch.Tensor:
    max_shift = int(SR * shift_ms / 1000)
    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return wav
    return torch.roll(wav, shifts=shift, dims=0)

def random_gain(wav: torch.Tensor, min_gain: float = 0.8, max_gain: float = 1.2) -> torch.Tensor:
    g = random.uniform(min_gain, max_gain)
    return wav * g

def mix_background(wav: torch.Tensor, bg_wav: torch.Tensor, snr_db_min=5.0, snr_db_max=20.0) -> torch.Tensor:
    """
    Mix wav with bg_wav at random SNR.
    Both are (T,) and 1s already.
    """
    snr_db = random.uniform(snr_db_min, snr_db_max)

    sig_power = wav.pow(2).mean().clamp_min(1e-8)
    noise_power = bg_wav.pow(2).mean().clamp_min(1e-8)

    # scale noise to target SNR: snr = 10*log10(sig/noise)
    target_noise_power = sig_power / (10 ** (snr_db / 10.0))
    scale = (target_noise_power / noise_power).sqrt()
    mixed = wav + bg_wav * scale

    return mixed.clamp(-1.0, 1.0)
