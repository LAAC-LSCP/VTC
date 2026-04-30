---
icon: lucide/home
---

# Voice Type Classifier (VTC)

The Voice Type Classifier is a free, open-source tool that listens to long audio recordings made around young children and labels *who is speaking when* — the child wearing the recorder, other children, adult women, or adult men.

---

<!-- <iframe
  id="audio-annotations"
  src="/assets/visualisations/audio-annotations-player.html"
  width="100%" height="520"
  style="border:1px solid var(--md-default-fg-color--lightest);border-radius:8px;display:block"
  loading="lazy"></iframe> -->

## Quick Start

VTC takes long, naturalistic audio recordings collected by a small device worn by a child (typically aged 0–5) and produces a timeline indicating when speech occurs and which broad category of speaker is talking: the **key child** wearing the recorder, **other children** nearby, an **adult female** voice, or an **adult male** voice. The input is one or more **WAV audio files** at 16 kHz, mono; the output is a spreadsheet (CSV) and a set of timestamp files (RTTM) listing every detected speech segment with a start time, duration, and speaker label. 

VTC does **not** transcribe what was said, and it does **not** distinguish between two different speakers of the same type (e.g., it cannot tell two adult women apart). Accuracy varies by speaker category — detections of the key child and adult female speech are most reliable, while detections of *other children* are the weakest and should be interpreted with care. VTC is **not a replacement for LENA**: it is a different system with similar but non-identical categories, and the two are not interchangeable in analyses.

If this matches what you need, head to the [Getting Started](getting-started.md) page to install VTC and run it on your first recording.

---

## What the speaker labels mean

> **The four categories VTC uses, in plain terms.**

VTC sorts every moment of detected speech into one of four buckets:

| Label | Meaning |
|-------|---------|
| **KCHI** | The "key child" — the child actually wearing the recorder. |
| **OCH**  | Any other child in the recording (siblings, playmates, classmates). |
| **MAL**  | An adult male voice. |
| **FEM**  | An adult female voice. |

These categories are **speaker types**, not individual people. VTC will tell you "an adult female spoke between 3.30s and 5.28s," but it cannot tell you *which* adult female if more than one is present. If two siblings are both in the room, both will be labelled OCH and VTC won't separate them.

---

## What VTC is *not*

> **Three common misconceptions, addressed up front, so you don't build an analysis on the wrong assumption.**

- **VTC is not a speech recognizer.** It identifies *who* is speaking *when*. It does not produce any transcript of *what* was said.
- **VTC is not a speaker diarizer.** Diarization tools try to track individual people across a recording (e.g., "Speaker A," "Speaker B"). VTC only assigns a category (KCHI / OCH / MAL / FEM). Two different adult females will both be labelled FEM with no distinction between them.
- **VTC is not a LENA replacement.** Both VTC and the commercial LENA system label speakers in child-centred recordings, but they are different systems with different categories, different recorders, and different licensing. VTC is free, open-source, runs on any Unix machine, accepts ordinary WAV files, and can be [fine-tuned](advanced/finetuning.md) on your own data. LENA is a commercial product that requires its own hardware recorder. Their speaker categories are similar but not identical (VTC uses KCHI/OCH/MAL/FEM, LENA uses CHN/CXN/MAN/FAN among others). For detailed accuracy comparisons, see the [ExELang book](https://bookdown.org/alecristia/exelang-book/accuracy.html), and a side-by-side comparison is on the [Version History](misc/versions.md#vtc-vs-lena) page.

---

## How accurate is VTC?

> **VTC's labels are estimates, not ground truth — and accuracy is uneven across the four speaker categories. Always interpret results with the error rates in mind.**

The table below shows how often VTC agrees with human annotators across the four categories. Higher numbers are better; the arrows (↑) indicate this in each column.

The score reported is an **F1 score**, which is a single number summarising two kinds of mistake: missing speech that was actually there ("false negatives") and labelling speech that wasn't there or was the wrong category ("false positives"). An F1 of 100% would mean perfect agreement with the human annotator; 0% would mean no agreement at all. The **Average F1** column is the simple mean across the four speaker types.

The bottom row, **Human 2**, shows how well a *second* human annotator agrees with the *first* human annotator on the same recordings. This is included as a reference point: even two careful human listeners do not perfectly agree, so this row is the practical ceiling we can hope for from any automated system.

| Model | KCHI ↑ | OCH ↑ | MAL ↑ | FEM ↑ | Average F1 ↑ |
|-------|------|-----|-----|-----|------------|
| VTC 1.0 | 68.2% | 30.5% | 41.2% | 63.7% | 50.9% |
| VTC 1.5 | 68.4% | 20.6% | 56.7% | 68.9% | 53.6% |
| **VTC 2.0** | **71.8%** | 51.4% | 60.3% | 74.8% | 64.6% |
| **VTC 2.1** | 67.1% | **56.1%** | **68.8%** | **75.5%** | **66.9%** |
| Human 2 | 79.7% | 60.4% | 67.6% | 71.5% | 69.8% |

**Reading the table in plain terms:**

- **Adult female (FEM)** and **the key child (KCHI)** are detected most reliably — roughly comparable to human-vs-human agreement.
- **Adult male (MAL)** detection has improved substantially across versions but still lags behind FEM. This may partly reflect the relative scarcity of adult male speech in many child-centred recordings.
- **Other children (OCH)** is the weakest category. Even the best version of VTC misses or mislabels a substantial share of these segments. **Treat OCH counts as rough estimates only**, especially when comparing recordings or aggregating across long sessions.

!!! warning "VTC outputs are estimates"
    Because the model makes errors, every count, rate, or duration you derive from VTC's output inherits some of that error. When two recordings differ by, say, 5% in adult-female speech minutes, that difference may or may not be real — you should always compare it against the model's known error margins, especially for the OCH category. Consider reporting confidence intervals or sensitivity analyses in published work.

!!! note "Where do these numbers come from?"
    These F1 scores are measured on the **BabyTrain-2025 validation set**, the standard benchmark used by the VTC project. Performance on your own recordings will vary depending on factors such as recording environment, languages spoken, ambient noise, and the age of the children. If you have annotated data of your own, you can [fine-tune VTC](advanced/finetuning.md) to improve accuracy on your specific population.

!!! tip "If accuracy on OCH or MAL matters a lot for your study"
    See [Threshold selection](advanced/thresholds.md) — VTC supports a "high precision" mode that trades off some recall (catching every segment) for higher precision (fewer wrong labels). This can be useful if you would rather miss some speech than risk introducing false detections.

---

## Related tools

> **VTC fits into a small ecosystem of tools for processing child-centred recordings; you may want some of these for your full analysis pipeline.**

| Tool | Role |
|------|------|
| [segma](https://github.com/arxaqapi/segma) | Audio segmentation library used to train and run VTC 2.0. |
| [BabyHuBERT](https://github.com/LAAC-LSCP/BabyHuBERT) | Self-supervised speech model on which VTC 2.0 is built. |
| [ALICE](https://github.com/orasanen/ALICE) | Adult word-count estimator that takes VTC output as input. |
| [ChildProject](https://childproject.readthedocs.io/) | Dataset management framework for child-centred recordings. |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | Speaker-processing toolkit; VTC borrows its segment-merging logic. |
| [vtc-finetune](https://github.com/arxaqapi/vtc-finetune) | Tools for fine-tuning VTC on your own annotated data. |

---

## Technical Reference

> *This section is intended for engineers and ML practitioners. Non-technical readers can stop here.*

### Evaluation methodology

- **Benchmark:** BabyTrain-2025 validation set.
- **Metric:** Per-class F1 over speaker activity detection, with the standard pyannote-style frame-level scoring. The Average F1 column is the unweighted macro-average over the four classes (KCHI, OCH, MAL, FEM).
- **"Human 2" row:** Inter-annotator agreement, computed by treating a second human annotator's labels as predictions against the first annotator's labels on the same evaluation portion of BabyTrain-2025. It is intended as a soft upper bound for what any model can be expected to achieve under the same labelling protocol.
- **Default decision thresholds** are tuned to maximise per-class F1 on the BabyTrain-2025 validation split (`f1.toml`); a high-precision threshold set (`hp.toml`) is also shipped and selected by maximising precision under a recall ≥ 0.5 constraint per class.

### Lineage

| | VTC 1.0 (2020) | VTC 1.5 (2025) | VTC 2.0 (2025) | VTC 2.1 (2025) |
|---|---|---|---|---|
| Backbone | PyanNet | Whisper-based | BabyHuBERT | BabyHuBERT |
| Default threshold set | per-class F1 | per-class F1 | per-class F1 (`f1.toml`) | per-class F1 (`f1.toml`) |
| Repository | [MarvinLvn/voice-type-classifier](https://github.com/MarvinLvn/voice-type-classifier) | [LAAC-LSCP/VTC-IS-25](https://github.com/LAAC-LSCP/VTC-IS-25) | [LAAC-LSCP/VTC](https://github.com/LAAC-LSCP/VTC) | [LAAC-LSCP/VTC](https://github.com/LAAC-LSCP/VTC) |

See the [Version History](misc/versions.md) page for label-set differences and migration notes from VTC 1.0.
