# Results Summary

Validation accuracy on the development split is summarized below.

| System | Inference | D2 after D3 | D3 | Avg |
|---|---:|---:|---:|---:|
| S1 | Soft MoE, tau=3.0 | 80.28 | 54.47 | 67.37 |
| S2 | Soft MoE, tau=4.0 | 80.13 | 54.71 | 67.42 |
| S3 | Soft MoE + TTA | 79.81 | 56.08 | 67.95 |
| S4 | Mean probability average | 79.81 | 55.96 | 67.88 |

D2 remains around 80% after D3 training. D3 is more difficult and has lower absolute accuracy. S3 obtains the best average validation accuracy. Mean probability averaging is competitive, suggesting entropy alone is not always a reliable routing signal.

Additional small result tables are provided under `docs/results/` and report-ready LaTeX snippets under `docs/report_tables/`.
