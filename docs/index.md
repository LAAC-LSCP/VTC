---
icon: lucide/home
---

# Voice Type Classifier (VTC)

The Voice Type Classifier is a deep learning model that identifies **who is speaking when** in child-centered long-form audio recordings. It detects speech segments and classifies them into one of four speaker categories:

| Label | Meaning |
|-------|---------|
| **KCHI** | Key child (wearing the recorder) |
| **OCH** | Other children (siblings, ...) |
| **MAL** | Adult male speech |
| **FEM** | Adult female speech |

VTC is designed for naturalistic recordings that are captured by a portable recorder worn by a child (typically 0 to 5 years old), often spanning several hours.

## What VTC is *not*

- **Not a speech recognizer** — it identifies *who* speaks *when*, not *what* they say.
- **Not a speaker diarizer** — it classifies speaker *types*, not individual identities (it cannot distinguish two adult females from each other).
- **Not a LENA replacement** — while VTC and LENA both classify speakers in child-centered recordings, they are different systems. VTC is free and open-source, runs on any Unix machine with any WAV audio, and its code and model weights are publicly available. LENA is a commercial, proprietary system that requires its own hardware recorder. Their speaker categories are similar but not identical (e.g., VTC uses KCHI/OCH/MAL/FEM while LENA uses CHN/CXN/MAN/FAN among others). VTC can be [fine-tuned](advanced/finetuning.md) to your data; LENA cannot. For detailed accuracy comparisons, see the [ExELang book](https://bookdown.org/alecristia/exelang-book/accuracy.html). A full comparison table is available on the [Version History](misc/versions.md#vtc-vs-lena) page.

---

## Model Accuracy

The table below shows how well each version of VTC performs at detecting each speaker type. Performance is measured using **F1 score**, a standard metric that combines two things: how often the model correctly detects a speaker type (recall) and how often its detections are actually correct (precision). An F1 score of 100% would mean perfect performance; higher values are better, as indicated by the ↑ arrows in the column headers.

Each column corresponds to one of the aformentioned speaker labels and an additional **Average F1** column is added, the average F1 score across all four speaker types.

The last row, "Human 2", shows the performance of a second human annotator compared to a first one. This gives a sense of how well even humans agree on these labels, and serves as an upper bound for what we can expect from an automated model.

| Model | KCHI ↑ | OCH ↑ | MAL ↑ | FEM ↑ | Average F1 ↑ |
|-------|------|-----|-----|-----|------------|
| VTC 1.0 | 68.2% | 30.5% | 41.2% | 63.7% | 50.9% |
| VTC 1.5 | 68.4% | 20.6% | 56.7% | 68.9% | 53.6% |
| **VTC 2.0** | **71.8%** | 51.4% | 60.3% | 74.8% | 64.6% |
| **VTC 2.1** | 67.1% | **56.1%** | **68.8%** | **75.5%** | **66.9%** |
| Human 2 | 79.7% | 60.4% | 67.6% | 71.5% | 69.8% |

**KCHI** and **FEM** are the most reliable classes. **OCH** is the weakest — use other-child counts with caution.

!!! warning "VTC outputs are estimates"
    Classification errors propagate into downstream analyses. Always account for error margins when interpreting results.

---

## Related tools

| Tool | Role |
|------|------|
| [segma](https://github.com/arxaqapi/segma) | Audio segmentation library powering VTC 2.0's training and inference |
| [BabyHuBERT](https://github.com/LAAC-LSCP/BabyHuBERT) | Self-supervised speech model that VTC 2.0 is built on |
| [ALICE](https://github.com/orasanen/ALICE) | Adult word count estimator — uses VTC output as input |
| [ChildProject](https://childproject.readthedocs.io/) | Dataset management framework for child-centered recordings |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | Speaker diarization toolkit; VTC uses its segment merging logic |
| [vtc-finetune](https://github.com/arxaqapi/vtc-finetune) | Fine-tuning tools for VTC |
