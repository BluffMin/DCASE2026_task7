# DCASE 2026 Task 7 - Fine-Tuned Expert Aggregation

This repository contains the cleaned source code for our DCASE 2026 Challenge Task 7 submission on domain-agnostic incremental audio classification.

The submitted systems use full fine-tuned MCnn14 domain experts. The official D1 checkpoint is used only as the initialization source. A D2 expert is trained with device augmentation, a D3 expert is trained with gain-shift augmentation, and inference aggregates D2/D3 expert probabilities.

## Submitted Systems

| System | Inference | Test-time augmentation |
|---|---|---|
| S1 | Entropy soft MoE, tau=3.0 | No |
| S2 | Entropy soft MoE, tau=4.0 | No |
| S3 | Entropy soft MoE, tau=3.0 | Full-safe TTA |
| S4 | Mean probability averaging | No |

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── config_task7.py
├── domain_net.py
├── train_domain_aug_tta_routing_d1_router_fast.py
├── generate_fullfinetune_repro_artifacts.py
├── scripts/
│   └── run_task7_fullfinetune_repro.sh
└── docs/
    ├── method_summary.md
    ├── results_summary.md
    ├── results/
    └── report_tables/
```

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code expects PyTorch with CUDA support for full training runs.

## Required Data and Checkpoints

Datasets, evaluation audio, and model checkpoints are not included in this public repository. Obtain the official DCASE Task 7 resources separately.

Expected external paths can be supplied through environment variables:

```bash
export DATA_ROOT=/path/to/task7_data
export D1_CKPT=/path/to/checkpoint_D1.pth
export EVAL_ROOT=/path/to/evaluation/audio
```

`DATA_ROOT` should contain the official metadata and evaluation setup files, including `evaluation_setup/development_train.txt` and `evaluation_setup/development_test.txt`.

## Training D2/D3 Experts

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_task7_fullfinetune_repro.sh
```

The runner trains plain MCnn14 D2/D3 experts from the official D1 checkpoint:

- D2 training augmentation: device augmentation
- D3 training augmentation: gain-shift augmentation
- Objective: cross entropy
- Optimizer: AdamW with cosine learning-rate schedule

Use `SMOKE_TEST=1` for a one-epoch syntax and pipeline check without producing final submission artifacts:

```bash
SMOKE_TEST=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_task7_fullfinetune_repro.sh
```

## Generating Submission Artifacts

After checkpoints exist, the runner calls `generate_fullfinetune_repro_artifacts.py` to compute development metrics, report tables, and submission-format outputs. Generated outputs are written under `runs/`, which is intentionally ignored by git.

## Results

See [docs/results_summary.md](docs/results_summary.md) for the validation summary. Small CSV summaries are included under `docs/results/` for reproducibility at the report level.

## Notes

- Checkpoints (`*.pth`, `*.pt`, `*.ckpt`) are intentionally excluded.
- Official development/evaluation audio is intentionally excluded.
- Submission output CSV/ZIP files are intentionally excluded from this source-code repository.
- The official D1 checkpoint must be obtained from the DCASE Task 7 resources.

## Citation / Task Link

Please refer to the official DCASE Challenge Task 7 page for task rules, data access, and evaluation protocol.

## Contact

Seungmin Heo: [smheo@seoultech.ac.kr](mailto:smheo@seoultech.ac.kr)
