"""Model registry and runtime helpers shared by the daily and dekadal foundation-model runners."""

from __future__ import annotations

import importlib
import inspect

# model key -> (module, class). Adapters import their heavy dependencies lazily.
REGISTRY = {
    "chronos2": ("bwb.models.fm_adapters", "Chronos2"),
    "chronos_bolt": ("bwb.models.fm_adapters", "ChronosBolt"),
    "timesfm25": ("bwb.models.fm_adapters", "TimesFM25"),
    "toto2": ("bwb.models.fm_adapters", "Toto2"),
    "tirex": ("bwb.models.fm_adapters", "TiRex"),
    "moirai2": ("bwb.models.fm_adapters", "Moirai2"),
    "ttm_r2": ("bwb.models.fm_adapters", "TTMr2"),
    "sundial": ("bwb.models.fm_adapters", "Sundial"),
}


def adapter_kwargs(cls, **extra) -> dict:
    """Keep only the keyword arguments an adapter's constructor accepts (e.g. TimesFM's `context`, TTM's `freq`)."""
    params = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in extra.items() if k in params}


def load_adapter(key: str, device: str, batch_size: int, **extra):
    module, name = REGISTRY[key]
    cls = getattr(importlib.import_module(module), name)
    return cls(device=device, batch_size=batch_size, **adapter_kwargs(cls, **extra))


class PeakMemory:
    """Peak GPU memory allocated by PyTorch between `reset()` and `read()` (MiB); None without CUDA."""

    def __init__(self):
        try:
            import torch

            self.torch = torch if torch.cuda.is_available() else None
        except ImportError:
            self.torch = None
        self.device = self.torch.cuda.get_device_name(0) if self.torch else None

    def reset(self):
        if self.torch:
            self.torch.cuda.reset_peak_memory_stats()

    def read(self):
        return round(self.torch.cuda.max_memory_allocated() / 2**20, 1) if self.torch else None
