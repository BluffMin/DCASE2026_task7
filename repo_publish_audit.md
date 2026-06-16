# Repository Publish Audit

## Source and Target

- Local source project: `/workspace/DCASE_Finetuning`
- Clean publish worktree: `/workspace/DCASE2026_task7_publish`
- Remote repository: `https://github.com/BluffMin/DCASE2026_task7`
- Target branch: `main`
- Backup branch created locally from current remote main: `backup/before-dcase-finetuning-main`

## What Was Copied

- Core configuration: `config_task7.py`
- Model definition: `domain_net.py`
- Training pipeline: `train_domain_aug_tta_routing_d1_router_fast.py`
- Artifact/report generation pipeline: `generate_fullfinetune_repro_artifacts.py`
- Reproduction runner: `scripts/run_task7_fullfinetune_repro.sh`
- Public documentation: `README.md`, `docs/method_summary.md`, `docs/results_summary.md`
- Small report-level result summaries: `docs/results/*.csv`
- Report-ready LaTeX snippets: `docs/report_tables/*.tex`
- Dependency list: `requirements.txt`
- Strong public `.gitignore`

## What Was Excluded

- Raw datasets and `task7_data/`
- Evaluation audio and local evaluation directories
- Checkpoints and weight dictionaries: `*.pth`, `*.pt`, `*.ckpt`
- Final submission ZIP files
- Submission output CSV files
- Full `runs/`, `logs/`, `outputs/`, `wandb/`, and temporary audit folders
- Python caches and IDE metadata
- Large binary files
- Secrets or credentials

## Checks Run

- `python -m py_compile config_task7.py domain_net.py train_domain_aug_tta_routing_d1_router_fast.py generate_fullfinetune_repro_artifacts.py`
- `bash -n scripts/run_task7_fullfinetune_repro.sh`
- Basic Python import check for key modules
- `find . -type f -size +20M`
- Search for forbidden extensions: `*.pth`, `*.pt`, `*.ckpt`, `*.wav`, `*.flac`, `*.zip`
- Secret scan: `rg -i 'api_key|token|password|secret' .`

## Check Results

- Syntax checks: passed
- Basic imports: passed
- Large files over 20 MB: none found
- Forbidden model/audio/archive files: none found
- Secret scan: no matches found

## Final File Tree

```text
.gitignore
README.md
config_task7.py
docs/method_summary.md
docs/report_tables/report_table_classwise_S1.tex
docs/report_tables/report_table_development_results.tex
docs/report_tables/report_table_system_configs.tex
docs/results/all_experiments_development_d2_d3_avg.csv
docs/results/all_systems_summary.csv
docs/results/step2_step3_domain_classwise_accuracy.csv
docs/results_summary.md
domain_net.py
final_file_tree.txt
generate_fullfinetune_repro_artifacts.py
requirements.txt
scripts/run_task7_fullfinetune_repro.sh
train_domain_aug_tta_routing_d1_router_fast.py
```

## Unresolved TODOs

- The public repository does not include DCASE data, official D1 checkpoint, evaluation audio, trained D2/D3 checkpoints, or final submission outputs by design.
- Users must obtain official DCASE Task 7 resources separately and set `DATA_ROOT`, `D1_CKPT`, and optionally `EVAL_ROOT` before running the full pipeline.

## Ready to Push

The cleaned source repository is ready for review and push after confirming the final tree.

Because `CONFIRM_PUSH=1` was not set during preparation, no push was performed.

Recommended push commands:

```bash
git push origin backup/before-dcase-finetuning-main
git push origin main
```
