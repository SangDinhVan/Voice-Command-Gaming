import torch
import numpy as np

def get_mel_filters(sr, n_fft, n_mels, fmin, fmax):
    """
    Tạo bộ lọc Mel thủ công.
    Chuyển đổi từ Hz -> Mel -> Hz và tạo các bộ lọc tam giác.
    """
    # 1. Chuyển đổi fmin, fmax sang Mel scale
    # Công thức: m = 2595 * log10(1 + f / 700)
    mel_min = 2595 * np.log10(1 + fmin / 700.0)
    mel_max = 2595 * np.log10(1 + fmax / 700.0)

    # 2. Tạo n_mels + 2 điểm trên thang Mel (đều nhau)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)

    # 3. Chuyển ngược lại sang Hz
    # Công thức: f = 700 * (10^(m / 2595) - 1)
    hz_points = 700 * (10**(mel_points / 2595.0) - 1)

    # 4. Tìm chỉ số (index) tương ứng trên FFT bin
    # bin = f * (n_fft + 1) / sr  <-- xấp xỉ
    # FFT frequencies: k * sr / n_fft
    # k = f * n_fft / sr
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    # 5. Tạo ma trận lọc (Filter Matrix) [n_fft//2 + 1, n_mels]
    filters = np.zeros((n_fft // 2 + 1, n_mels))

    for i in range(1, n_mels + 1):
        # Điểm bắt đầu, đỉnh, và kết thúc của tam giác
        f_m_minus = bin_points[i - 1]
        f_m = bin_points[i]
        f_m_plus = bin_points[i + 1]

        # Cạnh lên (Upslope)
        for k in range(f_m_minus, f_m):
            filters[k, i - 1] = (k - f_m_minus) / (f_m - f_m_minus)

        # Cạnh xuống (Downslope)
        for k in range(f_m, f_m_plus):
            filters[k, i - 1] = (f_m_plus - k) / (f_m_plus - f_m)

    return torch.tensor(filters, dtype=torch.float32)

class LogMelExtractor(torch.nn.Module):
    """
    Trích xuất đặc trưng bộ lọc Log-Mel thủ công.
    Thực hiện: Waveform -> STFT -> Power Spec -> Mel Filterbank -> Log
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
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = True

        # 1. Tạo Mel Filterbank Matrix thủ công
        mel_filters = get_mel_filters(sample_rate, n_fft, n_mels, f_min, f_max)
        # Đăng ký buffer để nó tự động di chuyển theo device (CPU/GPU) của model
        self.register_buffer('mel_filters', mel_filters)

        # 2. Tạo cửa sổ Hann (Hann Window)
        window = torch.hann_window(win_length)
        self.register_buffer('window', window)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (Batch, Time)
        returns: (Batch, n_mels, Time)
        """
        # Đảm bảo input là 2D (Batch, Time)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # 1. Short-Time Fourier Transform (STFT)
        # Output: (Batch, Freq, Time) complex
        # Note: torch.stft trả về complex tensor trong các bản pytorch mới
        stft_output = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            return_complex=True,
            pad_mode='reflect'
        )

        # 2. Power Spectrum (Năng lượng)
        # |STFT|^2, shape: (Batch, n_fft//2 + 1, Time)
        power_spec = stft_output.abs().pow(2.0)

        # 3. Áp dụng Mel Filterbank
        # power_spec: (B, F, T) -> transpose -> (B, T, F)
        # mel_filters: (F, n_mels)
        # Output: (B, T, n_mels)
        mel_spec = torch.matmul(power_spec.transpose(1, 2), self.mel_filters)

        # Transpose lại để có dạng (B, n_mels, T)
        mel_spec = mel_spec.transpose(1, 2)

        # 4. Logarithm
        logmel = torch.log(mel_spec + self.eps)

        # 5. Normalize (Instance Norm)
        if self.normalize:
            mean = logmel.mean(dim=(-2, -1), keepdim=True)
            std = logmel.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            logmel = (logmel - mean) / std

        return logmel