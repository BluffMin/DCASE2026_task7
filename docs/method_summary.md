# Method Summary

This submission addresses DCASE 2026 Task 7, domain-agnostic incremental audio classification, with full fine-tuned MCnn14 domain experts.

## Training

The official D1 checkpoint is used as the initialization source. We then train separate plain MCnn14 experts for D2 and D3. The D2 expert is fine-tuned on D2 training data with device augmentation. The D3 expert is fine-tuned on D3 training data with gain-shift augmentation. The training objective is cross entropy, optimized with AdamW and a cosine learning-rate schedule.

D1 is not used as a fallback or third inference expert in the submitted systems. The submitted systems aggregate only the D2 and D3 experts.

## Inference

Inference is performed at the probability level using D2/D3 expert outputs.

- S1: entropy soft mixture of experts with tau=3.0.
- S2: entropy soft mixture of experts with tau=4.0.
- S3: entropy soft mixture of experts with tau=3.0 plus full-safe waveform test-time augmentation averaging.
- S4: deterministic mean probability averaging of D2 and D3 experts.

The entropy soft MoE assigns smoother expert weights from entropy-derived confidence scores. The mean averaging baseline is included as a deterministic non-TTA comparison.
