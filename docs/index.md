---
icon: lucide/home
---

# Voice Type Classifier (VTC)

The Voice Type Classifier is a free, open-source tool that labels *who is speaking when* in long audio recordings of young children: the child wearing the recorder, other children, adult women, or adult men.

---

<!-- <iframe
  id="audio-annotations"
  src="/assets/visualisations/audio-annotations-player.html"
  width="100%" height="520"
  style="border:1px solid var(--md-default-fg-color--lightest);border-radius:8px;display:block"
  loading="lazy"></iframe> -->

## Quick Start

VTC processes long, naturalistic recordings made with a small device worn by a child (typically aged 0–5). For each recording, it finds when speech occurs and which kind of speaker is talking: the **key child** wearing the recorder, **other children**, an **adult female**, or an **adult male**. The input is one or more **WAV files** (16 kHz, mono). The output is a CSV file and a set of RTTM files listing every detected speech segment with its start time, duration and speaker label.

VTC does **not** transcribe speech, and it does **not** tell apart two speakers of the same type, such as two adult women. Accuracy depends on the speaker category: adult female and key child speech are detected most reliably, other children least reliably. VTC is **not a replacement for LENA**: the two systems use similar but different categories, and their results are not interchangeable.

To install VTC and run it on your first recording, go to [Getting Started](getting-started.md).

---

## Speaker labels

| Label | Meaning |
|-------|---------|
| **KCHI** | The key child, who wears the recorder. |
| **OCH**  | Any other child (siblings, playmates, classmates). |
| **MAL**  | An adult male voice. |
| **FEM**  | An adult female voice. |

These labels are **speaker types**, not individual people. VTC reports that an adult female spoke between 3.30 s and 5.28 s, but not *which* adult female if several are present. Two siblings in the same room are both labelled OCH.

---

## What VTC is *not*

- **A speech recognizer.** VTC finds *who* speaks *when*, not *what* is said.
- **A speaker diarizer.** Diarization tools track individual people across a recording ("Speaker A", "Speaker B"). VTC only assigns a category, so two different adult women are both labelled FEM.
- **A LENA replacement.** Both systems label speakers in child-centred recordings, but they differ in categories, hardware and licensing. VTC is free and open-source, runs on any Unix machine, accepts ordinary WAV files, and can be [fine-tuned](advanced/finetuning.md) on your own data. LENA is a commercial product that requires its own recorder. VTC uses KCHI/OCH/MAL/FEM; LENA uses CHN/CXN/MAN/FAN, among others. For accuracy comparisons, see the [ExELang book](https://bookdown.org/alecristia/exelang-book/accuracy.html); for a side-by-side comparison, see [Version History](misc/versions.md#vtc-vs-lena).

---

## How accurate is VTC?

The table below compares VTC's output with human annotations, as an F1 score per category (higher is better).

The F1 score combines two kinds of error: missed speech (false negatives) and speech detected where there was none or with the wrong label (false positives). 100% means perfect agreement with the human annotator. **Average F1** is the mean over the four categories.

The **Human 2** row compares a second human annotator with the first on the same recordings. Two human annotators do not fully agree either, so this row is a practical upper bound for an automated system.

| Model | KCHI ↑ | OCH ↑ | MAL ↑ | FEM ↑ | Average F1 ↑ |
|-------|------|-----|-----|-----|------------|
| VTC 1.0 | 68.2% | 30.5% | 41.2% | 63.7% | 50.9% |
| VTC 1.5 | 68.4% | 20.6% | 56.7% | 68.9% | 53.6% |
| VTC 2.0 | **71.8%** | 51.4% | 60.3% | 74.8% | 64.6% |
| VTC 2.1 | 67.1% | 56.1% | 68.8% | 75.5% | 66.9% |
| **VTC 2.2** | 68.5% | **58.1%** | **70.9%** | **75.6%** | **68.3%** |
| Human 2 | 79.7% | 60.4% | 67.6% | 71.5% | 69.8% |

- **Adult female (FEM)** speech is detected most reliably, slightly above the agreement between two human annotators.
- **Key child (KCHI)** detection is also reliable, but about 11 points below human agreement.
- **Adult male (MAL)** detection has improved a lot across versions and is now close to FEM. Adult male speech is relatively rare in many child-centred recordings.
- **Other children (OCH)** is the weakest category: VTC misses or mislabels a large share of these segments. **Treat OCH counts as rough estimates**, especially when comparing recordings or aggregating over long sessions.

!!! warning "VTC outputs are estimates"
    Every count, rate or duration derived from VTC's output inherits the model's errors. If two recordings differ by 5% in adult female speech, the difference may not be real. Compare such differences with the model's error rates, especially for OCH, and consider reporting confidence intervals or sensitivity analyses in published work.

!!! note "Where do these numbers come from?"
    The scores are measured on the **BabyTrain-2025 validation set**, the standard benchmark of the VTC project. Accuracy on your own recordings depends on the recording environment, the languages spoken, background noise and the age of the children. If you have annotated data, you can [fine-tune VTC](advanced/finetuning.md) on it.

!!! tip "If OCH or MAL accuracy matters for your study"
    VTC has a high-precision mode that produces fewer wrong labels but misses more speech. See [Threshold selection](advanced/thresholds.md).

---

## Related tools

| Tool | Role |
|------|------|
| [segma](https://github.com/arxaqapi/segma) | Audio segmentation library used to train and run VTC 2.x. |
| [BabyHuBERT](https://github.com/LAAC-LSCP/BabyHuBERT) | Self-supervised speech model on which VTC 2.x is built. |
| [ALICE](https://github.com/orasanen/ALICE) | Adult word-count estimator that takes VTC output as input. |
| [ChildProject](https://childproject.readthedocs.io/) | Dataset management framework for child-centred recordings. |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | Speaker-processing toolkit; VTC uses its segment-merging logic. |
| [vtc-finetune](https://github.com/arxaqapi/vtc-finetune) | Tools for fine-tuning VTC on your own annotated data. |

---

## Technical Reference

### Evaluation methodology

- **Benchmark:** BabyTrain-2025 validation set.
- **Metric:** Per-class F1 over speaker activity detection, with standard pyannote-style frame-level scoring. Average F1 is the unweighted macro-average over KCHI, OCH, MAL and FEM.
- **"Human 2" row:** Inter-annotator agreement, computed by scoring a second annotator's labels against the first annotator's labels on the same evaluation portion of BabyTrain-2025. It is a soft upper bound on what a model can achieve under the same labelling protocol.
- **Decision thresholds:** The default set (`f1.toml`) maximises per-class F1 on the BabyTrain-2025 validation split. The high-precision set (`hp.toml`) maximises precision under a recall ≥ 0.5 constraint per class. Both are published with the model on [Hugging Face](https://huggingface.co/coml/VTC-2).

### Lineage

| | VTC 1.0 (2020) | VTC 1.5 (2025) | VTC 2.0 (2025) | VTC 2.1 (2025) | VTC 2.2 (2026) |
|---|---|---|---|---|---|
| Backbone | PyanNet | Whisper-based | BabyHuBERT | BabyHuBERT | BabyHuBERT (2.1 weights, overlapping sliding-window inference) |
| Default threshold set | per-class F1 | per-class F1 | per-class F1 (`f1.toml`) | per-class F1 (`f1.toml`) | per-class F1 (`f1.toml`) |
| Repository | [MarvinLvn/voice-type-classifier](https://github.com/MarvinLvn/voice-type-classifier) | [LAAC-LSCP/VTC-IS-25](https://github.com/LAAC-LSCP/VTC-IS-25) | [LAAC-LSCP/VTC](https://github.com/LAAC-LSCP/VTC) | [LAAC-LSCP/VTC](https://github.com/LAAC-LSCP/VTC) | [LAAC-LSCP/VTC](https://github.com/LAAC-LSCP/VTC) |

See [Version History](misc/versions.md) for label-set differences and migration notes.
