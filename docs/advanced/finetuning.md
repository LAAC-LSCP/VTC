---
# icon: lucide/server-cog # gpu
icon: octicons/ai-model-24
---

# Fine-tuning VTC 2

> **If your recordings differ a lot from those VTC was trained on (e.g., a different language, an unusual recording environment, or a different age range), accuracy may drop. "Fine-tuning" is the process of giving VTC additional examples from your own setting so it can adapt. This page explains when fine-tuning is worth it and points engineers to the code.**

## Should you fine-tune?

Fine-tuning is **not** the first thing to try. It is a substantial undertaking — you will need:

- **Annotated audio**, where a human has gone through the recordings and marked, for each segment of speech, which speaker category it belongs to (KCHI / OCH / MAL / FEM). This is the part that takes the most time. As a rough guide, projects typically need at least a few hours of carefully annotated audio split across training and validation sets.
- **A computer with a powerful GPU** (48 GB of memory or more — a typical research-cluster card).
- **Familiarity with deep-learning training pipelines**, or someone on your team who has it. The fine-tuning code is open-source but assumes you can read and modify Python configuration files.

Before going down this route, consider easier alternatives:

1. Run VTC out of the box and look at the error patterns on a small annotated subset of your data. Errors may be acceptable for your research question.
2. Try the **high-precision threshold set** (see [Threshold selection](thresholds.md)) if false detections are your main concern.
3. Re-tune the **decision thresholds** on a small annotated set of your own data — this is much cheaper than full fine-tuning and often recovers a meaningful chunk of the accuracy gap.

If those steps aren't enough, fine-tuning is the next lever to pull.

!!! note "Expected accuracy gains"
    The size of the improvement you can expect from fine-tuning depends heavily on how different your data is from BabyTrain-2025 and on how much annotated data you have. There is no general guarantee of improvement; some setups see large gains, others see modest ones. Always evaluate your fine-tuned model against the base VTC 2 model on a held-out test set before relying on it.

---

## Technical Reference

> *Workflow and tooling for engineers carrying out fine-tuning.*

The fine-tuning code lives at [arxaqapi/vtc-finetune](https://github.com/arxaqapi/vtc-finetune).

The general workflow is:

1. **Prepare annotated data.** Audio (`.wav`) plus reference annotations (`.rttm`), split into train / validation / test partitions.
2. **Configure training** starting from the pre-trained VTC 2 checkpoint. Modify the config to point at your data splits and to any hyperparameter overrides.
3. **Train and evaluate.** Run training and compare against the base VTC 2 model on your held-out test set to confirm improvement before deploying.

See the `vtc-finetune` repository for installation instructions, configuration reference, and example scripts.
