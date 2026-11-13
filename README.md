# Multimodal cervical-imaging research

This project explores machine-learning methods for representation learning and classification using cervical-imaging datasets. The code covers preprocessing, augmentations, self-supervised strategies, encoders, training and evaluation scripts.

**Research software only.** It is not approved or validated for clinical diagnosis, screening, treatment or patient triage.

## Code structure

- `configs/` — configurable training parameters.
- `data/` — dataset loading and preprocessing.
- `models/` — model backbones, attention and fusion components.
- `scripts/` — training, evaluation and inference entry points.
- `training/` — self-supervised trainer implementation.
- `docs/` — proposals, flowcharts and GPU guidance.

## Environment

Use a Python virtual environment and the dependencies in `requirements.txt`. GPU acceleration depends on compatible PyTorch/CUDA builds and available hardware.

```bash
git clone https://github.com/BalajiKoushik01/Multimodal.git
cd Multimodal
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Training scripts include `scripts/train_ssl.py`, `scripts/train_finetune.py` and `scripts/train_worldclass.py`. Review their argument handling and `configs/config.yaml` before running. The latter is a project-local filename, **not evidence of exceptional results**.

## Data and evaluation

Datasets and checkpoints must be obtained separately with appropriate permissions. Any evaluation should document:

- the exact dataset version and inclusion criteria;
- patient-level splitting and possible data leakage;
- baseline comparisons and uncertainty intervals;
- external validation and failure cases.

Performance targets in historical planning documents are goals, not measured results. No clinical sensitivity, diagnostic accuracy or real-world impact is claimed without independently verified evidence.

## Evaluation reliability safeguards (November 2026)

The inference and evaluation scripts now **fail instead of producing predictions** when model checkpoints are absent or incompatible. Checkpoints are loaded in tensor-only mode and their state dictionaries must match the model. Corrupted/unreadable evaluation images raise an error rather than being silently replaced with black tensors. These are research-integrity improvements, **not** clinical validation or proof that an existing checkpoint is safe or accurate. Run the scripts only with trusted checkpoints and properly licensed, independently reviewed datasets.
