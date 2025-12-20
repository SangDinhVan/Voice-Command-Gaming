import torch
import torchaudio

class LogMelExtractor(torch.nn.Module):
    """
    Extract log-mel spectrogram from waveform (1s, 16kHz).
    Must be used identically for training and inference.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,          # 25ms
        win_length: int = 400,
        hop_length: int = 160,     # 10ms
        n_mels: int = 40,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        eps: float = 1e-10,
        normalize: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.normalize = normalize

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            center=True,
            pad_mode="reflect",
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (T,) or (1,T)
        returns: (1, n_mels, time)
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # (1,T)

        mel = self.melspec(wav)  # (1, n_mels, time)
        logmel = torch.log(mel + self.eps)

        if self.normalize:
            mean = logmel.mean(dim=(-2, -1), keepdim=True)
            std = logmel.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            logmel = (logmel - mean) / std

        return logmel
