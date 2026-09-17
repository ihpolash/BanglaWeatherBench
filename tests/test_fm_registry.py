import numpy as np
import pytest

from bwb.models.fm_base import FoundationModel, count_parameters
from bwb.models.fm_registry import REGISTRY, adapter_kwargs


class _WithContext(FoundationModel):
    def __init__(self, device="cpu", batch_size=4, context=1024):
        super().__init__(device, batch_size)
        self.context = context


def test_adapter_kwargs_passes_only_accepted_arguments():
    assert adapter_kwargs(_WithContext, context=2048, freq="W") == {"context": 2048}
    assert adapter_kwargs(FoundationModel, context=2048, freq="W") == {}


def test_registry_covers_the_week4_roster():
    assert set(REGISTRY) == {"chronos2", "chronos_bolt", "timesfm25", "toto2", "tirex", "moirai2", "ttm_r2", "sundial"}


def test_count_parameters_walks_nested_holders_and_dedupes_tied_weights():
    torch = pytest.importorskip("torch")
    shared = torch.nn.Linear(3, 2)  # 8 parameters

    class Predictor:  # e.g. a GluonTS predictor holding its network
        def __init__(self):
            self.prediction_net = torch.nn.Sequential(shared, torch.nn.Linear(2, 1))  # + 3

    class Adapter:
        def __init__(self):
            self._models = {30: Predictor()}
            self.pipe = type("Pipe", (), {})()
            self.pipe.model = shared  # tied: counted once
            self.arr = np.zeros(5)

    assert count_parameters(Adapter()) == 11
    assert count_parameters(object()) == 0
