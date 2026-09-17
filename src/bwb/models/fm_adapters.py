"""Zero-shot adapters for the Week-4 foundation-model roster. Heavy dependencies are imported lazily, so each
adapter runs in its own environment (see kaggle/fm_kernel). Output contract: `bwb.models.fm_base.FoundationModel`.

API facts verified 2026-09-15 (model cards / READMEs / local tests), see reports/week4_fm_report.md.
"""

from __future__ import annotations

import numpy as np

from bwb.models.fm_base import QUANTILES, FoundationModel, round_up_to

Q = [float(q) for q in QUANTILES]


def _torch():
    import torch

    return torch


class Chronos2(FoundationModel):
    """amazon/chronos-2 (Apache-2.0). predict_quantiles on a list of 1-D tensors; NaN accepted (verified locally)."""

    name, max_context, nan_policy = "chronos2", 8192, "native"

    def __init__(self, device="cuda", batch_size=256):
        super().__init__(device, batch_size)
        from chronos import BaseChronosPipeline

        self.pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=device)

    def _predict(self, contexts, horizon):
        torch = _torch()
        q, mean = self.pipe.predict_quantiles([torch.tensor(c) for c in contexts], prediction_length=horizon, quantile_levels=Q)
        q = torch.cat([x.reshape(1, horizon, len(Q)) for x in q]) if isinstance(q, list) else q.reshape(-1, horizon, len(Q))
        mean = torch.cat([m.reshape(1, horizon) for m in mean]) if isinstance(mean, list) else mean.reshape(-1, horizon)
        return mean.float().cpu().numpy(), q.permute(0, 2, 1).float().cpu().numpy()


class ChronosBolt(Chronos2):
    """amazon/chronos-bolt-base (Apache-2.0). Same call; returns a batched tensor (verified locally). Max context 2048."""

    name, max_context, nan_policy = "chronos_bolt", 2048, "native"

    def __init__(self, device="cuda", batch_size=256):
        FoundationModel.__init__(self, device, batch_size)
        from chronos import BaseChronosPipeline

        self.pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map=device)


class TimesFM25(FoundationModel):
    """google/timesfm-2.5-200m-pytorch (Apache-2.0). quantile_forecast[..., 0] is the mean, [..., 1:10] are q0.1..q0.9.
    NaN handling undocumented -> interpolated contexts. The compiled context equals the requested context length
    (1,024 in the main run); max_context + max_horizon (256) must stay within the model's 16,384-step limit
    (`context_limit` in timesfm_2p5_base.py). Shorter inputs are zero-padded and masked by timesfm (configs.py)."""

    name, max_context, nan_policy = "timesfm25", 16128, "interpolate"

    def __init__(self, device="cuda", batch_size=256, context=1024):
        super().__init__(device, batch_size)
        import timesfm

        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        # per_core_batch_size defaults to 1 in timesfm's ForecastConfig (verified in configs.py); left at 1 it forecast one
        # series at a time (~63 s per 256 windows on a T4 in smoke v3). Match it to the adapter batch.
        self.model.compile(timesfm.ForecastConfig(
            max_context=min(context, self.max_context), max_horizon=256, normalize_inputs=True, per_core_batch_size=batch_size,
            use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))

    def _predict(self, contexts, horizon):
        _, qf = self.model.forecast(horizon=horizon, inputs=[np.asarray(c, dtype=np.float64) for c in contexts])
        qf = np.asarray(qf)[:, :horizon, :]
        return qf[:, :, 0], qf[:, :, 1:10].transpose(0, 2, 1)


class Toto2(FoundationModel):
    """Datadog/Toto-2.0-313m (Apache-2.0). forecast({target, target_mask, series_ids}) -> (9, B, V, H) quantiles.
    Missing values passed through target_mask (native). Contexts are left-padded to a common length with mask=False.

    Toto reduces the context in patches of `patch_size`, so a context length that is not a multiple of it fails with an
    einops shape error ("can't divide axis of length 36 in chunks of 32" on the 36-dekad CHIRPS run). The common length
    is therefore rounded up; the extra left padding is masked as unobserved, exactly like short-context padding."""

    name, max_context, nan_policy, patch_size = "toto2", 1024, "native", 32

    def __init__(self, device="cuda", batch_size=128):
        super().__init__(device, batch_size)
        from toto2 import Toto2Model

        torch = _torch()
        self.dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = Toto2Model.from_pretrained("Datadog/Toto-2.0-313m").to(self.dev).eval()

    def _predict(self, contexts, horizon):
        torch = _torch()
        L = round_up_to(max(len(c) for c in contexts), self.patch_size)
        target = np.zeros((len(contexts), 1, L), dtype=np.float32)
        mask = np.zeros((len(contexts), 1, L), dtype=bool)
        for i, c in enumerate(contexts):
            ok = ~np.isnan(c)
            target[i, 0, L - len(c):] = np.where(ok, c, 0.0)
            mask[i, 0, L - len(c):] = ok
        with torch.no_grad():
            q = self.model.forecast(
                {"target": torch.tensor(target, device=self.dev), "target_mask": torch.tensor(mask, device=self.dev),
                 "series_ids": torch.zeros((len(contexts), 1), dtype=torch.long, device=self.dev)},
                horizon=horizon, decode_block_size=768, has_missing_values=bool((~mask).any()))
        q = torch.as_tensor(q).float().cpu().numpy()[:, :, 0, :horizon]  # (9, B, H)
        q = q.transpose(1, 0, 2)
        return q[:, 4, :], q


class TiRex(FoundationModel):
    """NX-AI/TiRex (NX-AI community license). backend='torch' because Kaggle T4 (compute capability 7.5) cannot run the
    xLSTM CUDA kernels. Output shapes undocumented -> normalised defensively to (B, 9, H)."""

    name, max_context, nan_policy = "tirex", 2048, "interpolate"

    def __init__(self, device="cuda", batch_size=256):
        super().__init__(device, batch_size)
        from tirex import load_model

        self.model = load_model("NX-AI/TiRex", device=device, backend="torch")

    def _predict(self, contexts, horizon):
        torch = _torch()
        L = max(len(c) for c in contexts)
        batch = torch.tensor(np.stack([np.pad(c, (L - len(c), 0), mode="edge") for c in contexts]), dtype=torch.float32)
        q, mean = self.model.forecast(context=batch, prediction_length=horizon)
        q, mean = torch.as_tensor(q).float().cpu().numpy(), torch.as_tensor(mean).float().cpu().numpy()
        if q.shape[-1] == len(Q):  # (B, H, 9)
            q = q.transpose(0, 2, 1)
        return mean.reshape(len(contexts), horizon), q.reshape(len(contexts), len(Q), horizon)


class Moirai2(FoundationModel):
    """Salesforce/moirai-2.0-R-small (CC-BY-NC-4.0). GluonTS predictor; NaN target values are treated as unobserved."""

    name, max_context, nan_policy = "moirai2", 1680, "native"

    def __init__(self, device="cuda", batch_size=256):
        super().__init__(device, batch_size)
        self.device = device
        self._models = {}

    def _forecaster(self, horizon, context_length):
        key = (horizon, context_length)
        if key not in self._models:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

            m = Moirai2Forecast(module=Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"),
                                prediction_length=horizon, context_length=context_length, target_dim=1,
                                feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
            self._models[key] = m.create_predictor(batch_size=self.batch_size, device=self.device)
        return self._models[key]

    def _predict(self, contexts, horizon):
        import pandas as pd

        L = max(len(c) for c in contexts)
        predictor = self._forecaster(horizon, min(L, self.max_context))
        data = [{"target": np.asarray(c, dtype=np.float32), "start": pd.Period("2000-01-01", freq="D")} for c in contexts]
        fcs = list(predictor.predict(data))
        q = np.stack([np.stack([f.quantile(level)[:horizon] for level in Q]) for f in fcs])  # (B, 9, H)
        mean = np.stack([np.asarray(f.mean)[:horizon] if hasattr(f, "mean") else f.quantile(0.5)[:horizon] for f in fcs])
        return mean, q


class TTMr2(FoundationModel):
    """ibm-granite/granite-timeseries-ttm-r2 (Apache-2.0). For daily data only the r2.1 branches apply
    (512-96/512-48/360-60/180-60/90-30/52-16, verified from the HF repo refs 2026-09-15), so context is capped at 512;
    get_model(freq="D") picks the variant and truncates its output to the requested horizon.
    Point forecasts only: all quantile levels equal the point forecast (CRPS reduces to absolute error; excluded from
    calibration comparisons). Inputs standard-scaled per context, as the model card requires.
    `freq` must be a key of tsfm's DEFAULT_FREQUENCY_MAPPING (min ... D, W); any resolution of a day or longer selects the
    r2.1 branches (get_model.py). Dekadal (10-day) data has no mapping and is run with freq="W"."""

    name, max_context, nan_policy, probabilistic = "ttm_r2", 512, "interpolate", False

    def __init__(self, device="cuda", batch_size=512, freq="D"):
        super().__init__(device, batch_size)
        from tsfm_public.toolkit.get_model import get_model

        torch = _torch()
        self.dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self._get_model, self._models, self.freq = get_model, {}, freq

    def _model(self, horizon):
        if horizon not in self._models:
            m = self._get_model("ibm-granite/granite-timeseries-ttm-r2", context_length=self.max_context,
                                prediction_length=horizon, freq=self.freq)
            self._models[horizon] = m.to(self.dev).eval()
        return self._models[horizon]

    def _freq_token(self, model, batch):
        """Frequency-prefix-tuned ('ft') variants expect a resolution token, looked up in tsfm's mapping."""
        torch = _torch()
        if not getattr(model.config, "resolution_prefix_tuning", False):
            return None
        from tsfm_public.toolkit.time_series_preprocessor import DEFAULT_FREQUENCY_MAPPING

        idx = DEFAULT_FREQUENCY_MAPPING[self.freq]
        return torch.full((batch,), int(idx), dtype=torch.long, device=self.dev)

    def _predict(self, contexts, horizon):
        torch = _torch()
        L = self.max_context
        x = np.stack([np.pad(c[-L:], (max(0, L - len(c)), 0), mode="edge") for c in contexts]).astype(np.float32)
        mu, sd = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True) + 1e-6
        model = self._model(horizon)
        kwargs = {"past_values": torch.tensor((x - mu) / sd, device=self.dev)[:, :, None]}
        token = self._freq_token(model, len(contexts))
        if token is not None:
            kwargs["freq_token"] = token
        with torch.no_grad():
            out = model(**kwargs)
        pred = out.prediction_outputs.float().cpu().numpy()[:, :horizon, 0] * sd + mu
        return pred, np.repeat(pred[:, None, :], len(Q), axis=1)


class Sundial(FoundationModel):
    """thuml/sundial-base-128m (Apache-2.0); needs transformers==4.40.1 (isolated env). Generative: quantiles and mean
    are computed from sampled trajectories. NaN handling undocumented -> interpolated contexts."""

    name, max_context, nan_policy = "sundial", 2880, "interpolate"

    def __init__(self, device="cuda", batch_size=64, num_samples=100):
        super().__init__(device, batch_size)
        from transformers import AutoModelForCausalLM

        torch = _torch()
        self.dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained("thuml/sundial-base-128m", trust_remote_code=True).to(self.dev).eval()
        self.num_samples = num_samples

    def _predict(self, contexts, horizon):
        torch = _torch()
        L = max(len(c) for c in contexts)
        seqs = torch.tensor(np.stack([np.pad(c, (L - len(c), 0), mode="edge") for c in contexts]), dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            out = self.model.generate(seqs, max_new_tokens=horizon, num_samples=self.num_samples)
        s = torch.as_tensor(out).float().cpu().numpy().reshape(len(contexts), self.num_samples, -1)[:, :, :horizon]
        return s.mean(axis=1), np.quantile(s, Q, axis=1).transpose(1, 0, 2)
