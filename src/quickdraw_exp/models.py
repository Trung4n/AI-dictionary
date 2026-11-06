"""The two baselines. Both end in a classifier over 100 classes."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False), nn.BatchNorm2d(channels))
        self.skip = nn.Identity() if stride == 1 and in_channels == channels else nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, stride, bias=False), nn.BatchNorm2d(channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.body(x) + self.skip(x))


class RasterResNet(nn.Module):
    """ResNet-18 layout (He et al., 2016) narrowed for 64x64 single-channel input."""

    input_kind = "raster"

    def __init__(self, n_classes: int = 100, width: int = 32, blocks=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        stages, channels = [], width
        for i, count in enumerate(blocks):
            out = width * 2 ** i
            for j in range(count):
                stages.append(ResBlock(channels, out, 2 if i > 0 and j == 0 else 1))
                channels = out
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, n_classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(images)))


class CausalGRU(nn.Module):
    """A one-directional GRU: the output at a point depends only on earlier points."""

    input_kind = "sequence"

    def __init__(self, n_classes: int = 100, hidden: int = 256, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(inplace=True))
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_classes)

    def _pack(self, seq: torch.Tensor, lengths: torch.Tensor):
        return pack_padded_sequence(self.embed(seq), lengths.cpu(), batch_first=True, enforce_sorted=False)

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Logits after the last real point of each drawing."""
        _, hidden = self.gru(self._pack(seq, lengths))
        return self.head(hidden[-1])

    def forward_all_steps(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Logits after every point, ``(batch, max(lengths), classes)``."""
        out, _ = self.gru(self._pack(seq, lengths))
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.head(out)


def build_model(model_cfg: dict, n_classes: int = 100) -> nn.Module:
    kind = model_cfg["kind"]
    if kind == "cnn":
        return RasterResNet(n_classes, model_cfg["width"], tuple(model_cfg["blocks"]))
    if kind == "gru":
        return CausalGRU(n_classes, model_cfg["hidden"], model_cfg["layers"], model_cfg["dropout"])
    raise ValueError(f"unknown model kind {kind!r}")


def to_inputs(kind: str, batch, device, non_blocking: bool = False) -> tuple[tuple, torch.Tensor]:
    """A DataLoader batch as (model inputs, labels)."""
    if kind == "raster":
        images, labels = batch
        inputs = (images.float().div(255).unsqueeze(1).to(device, non_blocking=non_blocking),)
    elif kind == "sequence":
        seq, lengths, labels = batch
        inputs = (seq.float().to(device, non_blocking=non_blocking), lengths.long())
    else:
        raise ValueError(f"unknown input kind {kind!r}")
    return inputs, labels.long().to(device, non_blocking=non_blocking)
