---
icon: lucide/git-pull-request-create-arrow
---

# Version History

> **A short history of VTC versions, the differences between them, and what changes if you're migrating from an older version. Also a comparison with the LENA system, which is often used in the same research field.**

## VTC versions

> **Each version is a re-trained model; the most recent ones use a stronger underlying speech model and reach noticeably higher accuracy.**

| | VTC 1.0 (2020) | VTC 1.5 (2025) | VTC 2.0 (2025) | VTC 2.1 (2025) |
|---|---------|---------|---------|---------|
| **Underlying speech model** | PyanNet | Whisper-based | BabyHuBERT | BabyHuBERT |
| **Average F1** (BabyTrain-2025 validation set) | 50.9% | 53.6% | 64.6% | **66.9%** |
| **Speaker labels** | CHI(KCHI), OCH, MAL, FEM, SPEECH | KCHI, OCH, MAL, FEM | KCHI, OCH, MAL, FEM | KCHI, OCH, MAL, FEM |
| **Python version** | 3.7+ (conda) | 3.13+ (uv) | 3.13+ (uv) | 3.13+ (uv) |
| **Repository** | [MarvinLvn/voice-type-classifier](https://github.com/MarvinLvn/voice-type-classifier) | [LAAC-LSCP/VTC-IS-25](https://github.com/LAAC-LSCP/VTC-IS-25) | [LAAC-LSCP/VTC](https://github.com/LAAC-LSCP/VTC) | [LAAC-LSCP/VTC](https://github.com/LAAC-LSCP/VTC) |

VTC 2.x is built on [BabyHuBERT](https://github.com/LAAC-LSCP/BabyHuBERT), a self-supervised speech model trained specifically on child-centred audio, which is the main reason for the accuracy jump from earlier versions. The largest gains over VTC 1.0 are on the **other-children (OCH)** and **adult-male (MAL)** categories — both more than 20 percentage points of F1 better in VTC 2.1 than in VTC 1.0 on BabyTrain-2025. (See the accuracy table on the [home page](../index.md#how-accurate-is-vtc) for the per-class numbers.)

!!! note "What "F1 = 66.9%" means in practice"
    F1 is a single summary number per category, averaged across the four. A higher number means the model agrees more often with human annotators; it does **not** translate directly into "right 66.9% of the time" or "wrong 33.1% of the time." For a richer breakdown, see the per-category numbers and discussion on the [home page](../index.md#how-accurate-is-vtc).

---

## Migrating from VTC 1.0

> **What you'll need to change in your scripts and pipelines if you're moving from the original VTC.**

- **Label set has changed.** The combined `CHI` label (which lumped KCHI + OCH together) and the `SPEECH` label no longer exist. If you depend on either, combine `KCHI` and `OCH` in your downstream scripts.
- **Output formats are unchanged** at the format level: VTC 2 still writes RTTM, and now additionally produces a CSV.
- **Fresh install required.** VTC 2 cannot run inside a VTC 1.0 conda environment. Follow the [Getting Started](../getting-started.md) page to set up a clean install with `uv`.

---

## VTC vs. LENA

> **VTC and LENA both label speakers in child-centred recordings, but they are different products with different licensing, hardware, and category schemes. They are not interchangeable.**

| | VTC 2 | LENA |
|---|---------|------|
| **Cost** | Free, open-source | Commercial |
| **Hardware** | Any Unix machine; runs on any WAV file | Requires a LENA-branded recorder |
| **Speaker classes** | KCHI, OCH, MAL, FEM | CHN, CXN, MAN, FAN (plus others) |
| **Transparency** | Code and model weights publicly available | Proprietary |
| **Input format** | Any WAV audio (16 kHz mono) | LENA `.its` files |
| **Customisable to your data** | Yes — via [fine-tuning](../advanced/finetuning.md) or threshold re-tuning | No |

VTC is **not** a drop-in replacement for LENA. The categories are similar in spirit but not equivalent, and counts produced by the two systems on the same audio will differ. If your study compares to or reuses LENA results, treat any cross-system comparison with care. For detailed accuracy comparisons between the two systems, see the [ExELang book](https://bookdown.org/alecristia/exelang-book/accuracy.html).
