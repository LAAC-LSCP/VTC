---
icon: lucide/git-pull-request-create-arrow
---

# Version History

This page lists the VTC versions, explains how to migrate from older ones, and compares VTC with LENA.

## VTC versions

| | VTC 1.0 (2020) | VTC 1.5 (2025) | VTC 2.0 (2025) | VTC 2.1 (2025) | VTC 2.2 (2026) |
|---|---------|---------|---------|---------|---------|
| **Underlying speech model** | PyanNet | Whisper-based | BabyHuBERT | BabyHuBERT | BabyHuBERT (same weights as 2.1) |
| **Average F1** (BabyTrain-2025 validation set) | 50.9% | 53.6% | 64.6% | 66.9% | **68.3%** |
| **Speaker labels** | CHI(KCHI), OCH, MAL, FEM, SPEECH | KCHI, OCH, MAL, FEM | KCHI, OCH, MAL, FEM | KCHI, OCH, MAL, FEM | KCHI, OCH, MAL, FEM |
| **Python version** | 3.7+ (conda) | 3.13+ (uv) | 3.13+ (uv) | 3.13+ (uv) | 3.13+ (uv) |
| **Repository** | [MarvinLvn/voice-type-classifier](https://github.com/MarvinLvn/voice-type-classifier) | [LAAC-LSCP/VTC-IS-25](https://github.com/LAAC-LSCP/VTC-IS-25) | [LAAC-LSCP/VTC](https://github.com/LAAC-LSCP/VTC) | [LAAC-LSCP/VTC](https://github.com/LAAC-LSCP/VTC) | [LAAC-LSCP/VTC](https://github.com/LAAC-LSCP/VTC) |

VTC 2.x is built on [BabyHuBERT](https://github.com/LAAC-LSCP/BabyHuBERT), a self-supervised speech model trained on child-centred audio, which explains most of the accuracy gain over earlier versions. The largest gains over VTC 1.0 are for other children (OCH) and adult males (MAL), both more than 25 F1 points higher in VTC 2.2. Per-class scores are on the [home page](../index.md#how-accurate-is-vtc).

VTC 2.2 uses the VTC 2.1 weights but processes each recording with overlapping sliding windows (75% overlap by default, set by `--stride_pct`) and averages the predictions of overlapping windows. This improves accuracy for every category and makes processing slower (see [How long will it take?](../guide.md#how-long-will-it-take)). VTC 2.2 is also the first version installed as a Python package that downloads the model from Hugging Face.

!!! note "What an F1 of 68.3% means"
    F1 summarises agreement with human annotators; the table shows its average over the four categories. It does **not** mean that VTC is right 68.3% of the time.

---

## Migrating from VTC 2.0 or 2.1

- **The model is downloaded automatically** from [Hugging Face](https://huggingface.co/coml/VTC-2) on the first run. `git-lfs` and `git clone --recurse-submodules` are no longer needed, and you can delete the `VTC-2/` folder. After pulling the new version, run `uv sync` once.
- **New commands** replace the scripts in `scripts/`:

    | Before | Now |
    |---|---|
    | `uv run scripts/infer.py ...` | `uv run vtc-infer ...` |
    | `sh scripts/run.sh` | `sh run.sh` |
    | `uv run scripts/convert.py --input <in> --output <out>` | `uv run convert-audios --wavs <in> --output <out>` |

- **Renamed options:**

    | Before | Now |
    |---|---|
    | `--high_precision` | `--high-precision` |
    | `--thresholds path/to/custom.toml` | `--thresholds-location path/to/custom.toml --thresholds-name custom` (see [Custom threshold files](../advanced/thresholds.md#custom-threshold-files)) |
    | `--checkpoint VTC-2/model/best.ckpt`, `--config VTC-2/model/config.toml` | `--checkpoint 2.2`, `--config 2.2` (the defaults), or a local path |

- **Earlier models:** `--checkpoint 2.1` loads the same weights as 2.2, but results still differ slightly from VTC 2.1 runs because of the sliding windows. The VTC 2.0 model cannot be selected by version number: it only runs from the `v2.0.2` tag of the repository (see [Using VTC 2.0](#using-vtc-20)).
- **The output folder** now also contains the threshold file used for the run (`f1.toml` or `hp.toml`).

### Using VTC 2.0

The current version of VTC cannot run the VTC 2.0 model. To use it, for example to reproduce published results, clone the repository at the `v2.0.2` tag, the last 2.0 release, in a separate folder. This version needs [`git-lfs`](https://git-lfs.com/) to download the model:

```bash
git lfs install
git clone --branch v2.0.2 --recurse-submodules https://github.com/LAAC-LSCP/VTC.git VTC-2.0
cd VTC-2.0
uv sync

uv run scripts/infer.py      \
    --wavs <audio_folder>    \
    --output <output_folder> \
    --device cpu
```

This version is run with `scripts/infer.py` and has different options from the current `vtc-infer`. See the README of the [`v2.0.2` tag](https://github.com/LAAC-LSCP/VTC/tree/v2.0.2) for its options.

---

## Migrating from VTC 1.0

- **Labels:** the `CHI` label (KCHI and OCH combined) and the `SPEECH` label no longer exist. To recover `CHI`, combine `KCHI` and `OCH` in your own scripts.
- **Output:** VTC 2 still writes RTTM files, and also writes a CSV.
- **Installation:** VTC 2 cannot run in a VTC 1.0 conda environment. Install it from scratch following [Getting Started](../getting-started.md).

---

## VTC vs. LENA

| | VTC 2 | LENA |
|---|---------|------|
| **Cost** | Free, open-source | Commercial |
| **Hardware** | Any Unix machine; runs on any WAV file | Requires a LENA-branded recorder |
| **Speaker classes** | KCHI, OCH, MAL, FEM | CHN, CXN, MAN, FAN (plus others) |
| **Transparency** | Code and model weights publicly available | Proprietary |
| **Input format** | Any WAV audio (16 kHz mono) | LENA `.its` files |
| **Customisable to your data** | Yes, with [fine-tuning](../advanced/finetuning.md) or threshold re-tuning | No |

VTC is **not** a drop-in replacement for LENA. Their categories are similar but not equivalent, and the two systems produce different counts on the same audio. Be careful when comparing VTC results with LENA results. For accuracy comparisons, see the [ExELang book](https://bookdown.org/alecristia/exelang-book/accuracy.html).
