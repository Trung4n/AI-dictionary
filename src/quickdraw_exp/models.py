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

    def __init__(self, n_classes: int = 100, width: int = 32, blocks=(2, 2, 2, 2), stem_stride: int = 1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, width, 3, stem_stride, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        stages, channels = [], width
        for i, count in enumerate(blocks):
            out = width * 2 ** i
            for j in range(count):
                stages.append(ResBlock(channels, out, 2 if i > 0 and j == 0 else 1))
                channels = out
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, n_classes))
        self.feature_dim = channels

    def features(self, images: torch.Tensor) -> torch.Tensor:
        return self.head[1](self.head[0](self.stages(self.stem(images))))

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.head[2](features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classify(self.features(images))


class CausalGRU(nn.Module):
    """A one-directional GRU: the output at a point depends only on earlier points."""

    input_kind = "sequence"

    def __init__(self, n_classes: int = 100, hidden: int = 256, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(inplace=True))
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_classes)
        self.feature_dim = hidden

    def _pack(self, seq: torch.Tensor, lengths: torch.Tensor):
        return pack_padded_sequence(self.embed(seq), lengths.cpu(), batch_first=True, enforce_sorted=False)

    def features(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """The GRU state after the last real point of each drawing."""
        _, hidden = self.gru(self._pack(seq, lengths))
        return hidden[-1]

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Logits after the last real point of each drawing."""
        return self.classify(self.features(seq, lengths))

    def forward_all_steps(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Logits after every point, ``(batch, max(lengths), classes)``."""
        out, _ = self.gru(self._pack(seq, lengths))
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.head(out)


class AnytimeNet(nn.Module):
    """A backbone plus an optional confidence head.

    The head estimates how much probability the classifier gives the true class
    (its "true class probability", after Corbiere et al., 2019) from the backbone's
    features and the top three probabilities. Its inputs are detached, so training
    it never changes the classifier: with or without the head, the class
    predictions come from the same network.
    """

    def __init__(self, backbone: nn.Module, confidence_head: bool = False, hidden: int = 64):
        super().__init__()
        self.backbone = backbone
        self.input_kind = backbone.input_kind
        self.confidence = nn.Sequential(nn.Linear(backbone.feature_dim + 3, hidden), nn.ReLU(inplace=True),
                                        nn.Linear(hidden, 1)) if confidence_head else None

    def forward(self, *inputs) -> torch.Tensor:
        return self.backbone(*inputs)

    def forward_all(self, *inputs) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Logits, and the confidence in [0, 1] (None without a head)."""
        features = self.backbone.features(*inputs)
        logits = self.backbone.classify(features)
        if self.confidence is None:
            return logits, None
        top3 = torch.softmax(logits.detach().float(), dim=1).topk(3, dim=1).values
        joined = torch.cat([features.detach().float(), top3], dim=1)
        return logits, torch.sigmoid(self.confidence(joined).squeeze(1))


def build_model(model_cfg: dict, n_classes: int = 100) -> AnytimeNet:
    kind = model_cfg["kind"]
    if kind == "cnn":
        backbone = RasterResNet(n_classes, model_cfg["width"], tuple(model_cfg["blocks"]), model_cfg.get("stem_stride", 1))
    elif kind == "gru":
        backbone = CausalGRU(n_classes, model_cfg["hidden"], model_cfg["layers"], model_cfg["dropout"])
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    return AnytimeNet(backbone, model_cfg.get("confidence_head", False))


def load_weights(model: AnytimeNet, state_dict: dict) -> None:
    """Load weights, including those saved before the backbone was wrapped
    (the leak_v1 runs), whose names have no ``backbone.`` prefix."""
    if not any(key.startswith("backbone.") for key in state_dict):
        state_dict = {f"backbone.{key}": value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)


def to_inputs(kind: str, batch, device, non_blocking: bool = False, views: int = 1) -> tuple[tuple, torch.Tensor]:
    """A DataLoader batch as (model inputs, labels). With ``views`` > 1 every
    drawing brings that many prefixes; they are laid out drawing by drawing."""
    if kind == "raster":
        images, labels = batch
        if views > 1:
            images = images.reshape(-1, *images.shape[2:])
        inputs = (images.float().div(255).unsqueeze(1).to(device, non_blocking=non_blocking),)
    elif kind == "sequence":
        seq, lengths, labels = batch
        if views > 1:
            seq, lengths = seq.reshape(-1, *seq.shape[2:]), lengths.reshape(-1)
        inputs = (seq.float().to(device, non_blocking=non_blocking), lengths.long())
    else:
        raise ValueError(f"unknown input kind {kind!r}")
    labels = labels.long()
    if views > 1:
        labels = labels.repeat_interleave(views)
    return inputs, labels.to(device, non_blocking=non_blocking)


def anytime_loss(logits: torch.Tensor, labels: torch.Tensor, confidence, views: int,
                 consistency_weight: float, confidence_weight: float) -> tuple[torch.Tensor, dict]:
    """Cross-entropy on every prefix, plus two optional terms.

    Consistency: for prefixes of one drawing in drawing order, the earlier
    prediction is pulled toward the later one, KL(later || earlier), with the later
    one held fixed. The later prefix carries more ink, so it acts as a teacher that
    still reflects what the drawing could become.

    Confidence: squared error between the head's output and the probability the
    classifier currently gives the true class.
    """
    logits = logits.float()
    ce = torch.nn.functional.cross_entropy(logits, labels)
    total, parts = ce, {"ce": ce.detach()}
    if views > 1 and consistency_weight > 0:
        z = logits.reshape(-1, views, logits.shape[1])
        teacher = torch.softmax(z[:, 1:], dim=2).detach()
        student = torch.log_softmax(z[:, :-1], dim=2)
        kl = (teacher * (torch.log(teacher.clamp_min(1e-8)) - student)).sum(dim=2).mean()
        total = total + consistency_weight * kl
        parts["kl"] = kl.detach()
    if confidence is not None and confidence_weight > 0:
        target = torch.softmax(logits.detach(), dim=1).gather(1, labels[:, None]).squeeze(1)
        conf_loss = torch.nn.functional.mse_loss(confidence.float(), target)
        total = total + confidence_weight * conf_loss
        parts["conf"] = conf_loss.detach()
    return total, parts
