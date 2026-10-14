---
# icon: lucide/server-cog # gpu
icon: octicons/ai-model-24
---

# Fine-tuning VTC 2

VTC can be less accurate on recordings that differ from its training data, for example in another language, recording environment or age range. Fine-tuning continues training the model on annotated recordings from your own setting.

## Should you fine-tune?

Try simpler options first. Fine-tuning requires:

- **Annotated audio**, in which a person has marked each speech segment with its category (KCHI, OCH, MAL or FEM). This takes the most time. Projects typically need at least a few hours of annotated audio, split into training and validation sets.
- **A GPU with at least 48 GB of memory**, as found on research clusters.
- **Experience with deep-learning training**, or someone on your team who has it. The fine-tuning code assumes you can read and edit Python configuration files.

Simpler alternatives:

1. Run VTC as is and check its errors on a small annotated subset of your data. They may be acceptable for your research question.
2. If false detections are the main problem, use the [high-precision thresholds](thresholds.md).
3. Re-tune the [decision thresholds](thresholds.md#threshold-tuning-workflow) on a small annotated set. This is much cheaper than fine-tuning and often recovers part of the accuracy gap.

!!! note "Expected accuracy gains"
    The gain from fine-tuning depends on how much your data differs from BabyTrain-2025 and on how much annotated data you have. It can be large or small. Always compare the fine-tuned model with the base VTC 2 model on a held-out test set before using it.

---

## Technical Reference

The fine-tuning code is at [arxaqapi/vtc-finetune](https://github.com/arxaqapi/vtc-finetune). The workflow is:

1. **Prepare annotated data:** audio (`.wav`) and reference annotations (`.rttm`), split into train, validation and test sets.
2. **Configure training** from the pre-trained VTC 2 checkpoint: point the config at your data splits and set any hyperparameter overrides.
3. **Train and evaluate** against the base VTC 2 model on your held-out test set.

See the `vtc-finetune` repository for installation, configuration and example scripts. To run the fine-tuned model, pass its checkpoint and config to `vtc-infer` with `--checkpoint` and `--config` (see [Command Line Interface Arguments](../guide.md#command-line-interface-arguments)).
