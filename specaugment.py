import torch
import torchaudio

class SpecAugment(torch.nn.Module):
    def __init__(self,
                 n_time_mask: int,
                 n_freq_mask: int,
                 time_mask_param: int = 50,
                 freq_mask_param: int = 16):
        super().__init__()

        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)

        self.n_time_mask = n_time_mask
        self.n_freq_mask = n_freq_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        for _ in range(self.n_time_mask):
            x = self.time_mask(x)

        for _ in range(self.n_freq_mask):
            x = self.freq_mask(x)

        return x