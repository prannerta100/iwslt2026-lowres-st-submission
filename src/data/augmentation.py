"""Audio augmentation for low-resource ST training."""

import random

import torch
import torchaudio


class AudioAugmentor:
    """Applies a chain of audio augmentations.

    Augmentations:
    - Speed perturbation (0.9x, 1.0x, 1.1x)
    - Additive noise injection
    - SpecAugment is applied separately at the feature level in training scripts
    """

    def __init__(
        self,
        speed_perturbation: bool = True,
        speed_factors: list[float] = None,
        noise_injection: bool = True,
        noise_snr_db: list[float] = None,
    ):
        self.speed_perturbation = speed_perturbation
        self.speed_factors = speed_factors or [0.9, 1.0, 1.1]
        self.noise_injection = noise_injection
        self.noise_snr_db = noise_snr_db or [10.0, 15.0, 20.0]

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if self.speed_perturbation:
            waveform = self._apply_speed_perturbation(waveform, sample_rate)
        if self.noise_injection and random.random() < 0.3:
            waveform = self._apply_noise(waveform)
        return waveform

    def _apply_speed_perturbation(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        factor = random.choice(self.speed_factors)
        if factor == 1.0:
            return waveform
        # Use torch-based resampling instead of sox (which requires libsox)
        try:
            orig_len = waveform.shape[-1]
            new_len = int(orig_len / factor)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            resampled = torch.nn.functional.interpolate(
                waveform.unsqueeze(0), size=new_len, mode='linear', align_corners=False
            ).squeeze(0)
            if resampled.shape[0] == 1:
                resampled = resampled.squeeze(0)
            return resampled
        except Exception:
            return waveform

    def _apply_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        snr_db = random.choice(self.noise_snr_db)
        noise = torch.randn_like(waveform)
        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (snr_linear * noise_power + 1e-8))
        return waveform + scale * noise


class SpecAugment:
    """SpecAugment applied to mel-spectrogram features.

    Applied during training after feature extraction.
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=freq_mask_param
        )
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=time_mask_param
        )
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to features of shape (n_mels, time) or (batch, n_mels, time)."""
        for _ in range(self.num_freq_masks):
            features = self.freq_masking(features)
        for _ in range(self.num_time_masks):
            features = self.time_masking(features)
        return features
