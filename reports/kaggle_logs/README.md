# Kaggle run logs

Original notebook logs of the GPU and Kaggle CPU runs behind the paper, compressed with xz
(`xz -dc <file> | python -m json.tool`). Each is Kaggle's JSON stream of stdout/stderr records.
They record install output (package versions), torch/CUDA builds, per-model timings and verification.

| File | Run | Notebook |
|---|---|---|
| test_fm_zero_shot | 8 foundation models, test split (main run) | kaggle/fm_kernel |
| test_fm_install_smoke3 | same notebook, same day; full per-model install record | kaggle/fm_kernel |
| test_fm_week4b | foundation-model follow-up runs (week 4b) | kaggle/fm_week4b |
| test_fm_gapmatched_ctx2048 | gap-matched contexts, 2,048-day context | kaggle/fm_gm2048 |
| test_neural, _gpu0, _gpu1 | NHITS, DLinear, PatchTST, test split | kaggle/neural_kernel |
| test_neural_gapmatched, _worker | neural controls with gap-matched contexts | kaggle/neural_gapmatched |
| test_autoarima_a, _b | AutoARIMA, test split | kaggle/arima_a, arima_b |
| sundial_check | Sundial dispersion diagnostic | kaggle/sundial_check |
| val_fm | 5 foundation models, validation split (Python 3.13, torch 2.11+cu130) | kaggle/fm_val |
| val_fm_b | Toto-2.0, Moirai-2, Sundial, validation split (Python 3.12, torch 2.10+cu128) | kaggle/fm_val_b |
| val_neural, _gpu0, _gpu1 | neural baselines, validation split | kaggle/neural_val |
| val_autoarima_a, _b | AutoARIMA, validation split | kaggle/arima_val_a, arima_val_b |

The logs contain no credentials; they were scanned before commit.
