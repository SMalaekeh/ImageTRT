# ImageTRT

Production-ready scaffold for **fast image inference/training with TensorRT and PyTorch**. 
This cleanup pack gives you:
- A clear repo layout and job-ready README
- Reproducible env (Conda + pip)
- CI (lint + unit tests) on GitHub Actions
- Pre-commit hooks (ruff, black)
- Issue/PR templates, CONTRIBUTING, and a minimal **model card**

> Author: **Sayedmorteza Malaekeh (Ali)**

## Quick Start
```bash
conda env create -f environment.yml
conda activate imagetrt
pip install -e .
pytest -q
```

## Inference (example)
```bash
python -m imagetrt.infer --weights weights/engine.plan --image sample.jpg --device cuda
```

## Training (example)
```bash
python -m imagetrt.train --data data/ --epochs 10
```

## Repo layout (suggested)
```
imagetrt/           # Python package (train.py, infer.py, utils/)
scripts/            # CLI helpers (export_to_trt.py, benchmark.py, profile_gpu.py)
tests/              # unit tests
docs/               # model card, diagrams
notebooks/          # experiments (optional)
weights/            # models/engines (gitignored)
```

## Results (fill this in)
- FPS (FP32/FP16/INT8): 
- Latency (p50/p95): 
- Accuracy/F1: 
- Hardware: 
- Dataset: 

See `docs/model_card.md` for the full details.
