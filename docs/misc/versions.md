---
icon: lucide/git-pull-request-create-arrow
---

# Version History

## VTC versions

| | VTC 1.0 (2020) | VTC 1.5 (2025) | VTC 2.0 (2025) |
|---|---------|---------|---------|
| **Architecture** | PyanNet | Whisper-based | BabyHuBERT |
| **Average F1** | 50.9% | 53.6% | **64.6%** |
| **Labels** | CHI(KCHI), OCH, MAL, FEM, SPEECH | KCHI, OCH, MAL, FEM | KCHI, OCH, MAL, FEM |
| **Python** | 3.7+ (conda) | 3.13+ (uv) | 3.13+ (uv) |
| **Repository** | [MarvinLvn/voice-type-classifier](https://github.com/MarvinLvn/voice-type-classifier) | [LAAC-LSCP/VTC-IS-25](https://github.com/LAAC-LSCP/VTC-IS-25) | [LAAC-LSCP/VTC](https://github.com/LAAC-LSCP/VTC) |

VTC 2 uses [BabyHuBERT](https://github.com/LAAC-LSCP/BabyHuBERT), a self-supervised model trained specifically on child-centered audio. The biggest accuracy gains are on OCH (+21 points) and MAL (+19 points vs. v1.0).

## Migrating from VTC 1.0

- The `CHI` label (combining KCHI + OCH) and `SPEECH` label no longer exist. Combine `KCHI` and `OCH` in your scripts if needed.
- Output format is the same (RTTM), with CSV additionally provided.
- VTC 2 requires a fresh install — it cannot run in a VTC 1.0 conda environment.

## VTC vs. LENA

| | VTC 2 | LENA |
|---|---------|------|
| **Cost** | Free, open-source | Commercial |
| **Hardware** | Any Unix machine | Requires LENA recorder |
| **Speaker classes** | KCHI, OCH, MAL, FEM | CHN, CXN, MAN, FAN, + others |
| **Transparency** | Code and weights available | Proprietary |
| **Input** | Any WAV audio | LENA .its files |
| **Customizable** | Yes ([fine-tuning](advanced.md#fine-tuning)) | No |

VTC is not a drop-in replacement for LENA. Categories are similar but not identical. See the [ExELang book](https://bookdown.org/alecristia/exelang-book/accuracy.html) for detailed accuracy comparisons.
