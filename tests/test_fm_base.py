import numpy as np
import pytest

from bwb.models.fm_base import QUANTILES, FoundationModel, interpolate_gaps, trim_leading_nan


def test_interpolate_gaps_linear_and_edges():
    x = np.array([np.nan, 1.0, np.nan, 3.0, np.nan], dtype=np.float32)
    assert np.allclose(interpolate_gaps(x), [1.0, 1.0, 2.0, 3.0, 3.0])
    assert np.allclose(interpolate_gaps(np.full(4, np.nan)), 0.0)


def test_trim_leading_nan_keeps_interior_gaps():
    x = np.array([np.nan, np.nan, 5.0, np.nan, 7.0], dtype=np.float32)
    out = trim_leading_nan(x)
    assert len(out) == 3 and out[0] == 5.0 and np.isnan(out[1])


class _Echo(FoundationModel):
    name, max_context, nan_policy = "echo", 8, "interpolate"

    def _predict(self, contexts, horizon):
        last = np.array([c[-1] for c in contexts], dtype=np.float32)
        mean = np.repeat(last[:, None], horizon, axis=1)
        q = np.stack([mean + d for d in (QUANTILES - 0.5)[::-1]], axis=1)  # deliberately unsorted
        return mean, q


def test_prepare_caps_context_and_policy():
    m = _Echo(batch_size=2)
    x = np.arange(20, dtype=np.float32)
    x[-2] = np.nan
    ctx = m.prepare(x, context_length=1024)
    assert len(ctx) == 8 and not np.isnan(ctx).any()


def test_predict_batch_shapes_sorted_and_batched():
    m = _Echo(batch_size=2)
    mean, q = m.predict_batch([np.ones(5), np.full(5, 2.0), np.full(5, 3.0)], horizon=4)
    assert mean.shape == (3, 4) and q.shape == (3, 9, 4)
    assert (np.diff(q, axis=1) >= 0).all()


def test_predict_batch_rejects_bad_shapes():
    class Bad(_Echo):
        def _predict(self, contexts, horizon):
            return np.zeros((len(contexts), horizon)), np.zeros((len(contexts), 3, horizon))

    with pytest.raises(ValueError):
        Bad().predict_batch([np.ones(3)], horizon=2)


def test_round_up_to_patch_multiples():
    from bwb.models.fm_base import round_up_to

    # Toto's patch size is 32: the 36/72/144-dekad CHIRPS contexts must be padded up before the model sees them.
    assert [round_up_to(n, 32) for n in (36, 72, 144, 1024)] == [64, 96, 160, 1024]
    assert round_up_to(1, 32) == 32 and round_up_to(64, 32) == 64
