# Voice Type Classifier (VTC) 2.2

The Voice Type Classifier identifies *who speaks when* in audio recordings.

The four classes that the model will output are:
- **FEM** stands for adult female speech
- **MAL** stands for adult male speech
- **KCHI** stands for key-child speech (the child wearing the microphone)
- **OCH** stands for other children speech

The model has been specifically trained to work with child-centered long-form recordings. These are recordings that can span multiple hours and have been collected using a portable recorder attached to the vest of a child (usually 0 to 5 years of age).

## 0. Table of contents

1. [Installation](#1-installation)
2. [Inference](#2-inference)
3. [Model Performance](#3-model-performance)
4. [Citation](#4-citation)
5. [Acknowledgement](#5-acknowledgement)

## 1. Installation

To use the model, you will need a unix-based machine (Linux or MacOS). Windows is not supported for the moment. You don't need to install Python yourself: uv installs Python 3.13 when you run `uv sync`.
As system dependencies, ensure that [uv](https://docs.astral.sh/uv/) and [ffmpeg](https://ffmpeg.org/) (version 4 to 8) are installed. FFmpeg 9 is not supported yet; on macOS, Homebrew's default `ffmpeg` is version 9, so follow the `ffmpeg@8` steps in the [documentation](https://laac-lscp.github.io/VTC/getting-started/#installation). You can check that the tools are installed by running:

```bash
./check_sys_dependencies.sh
```

You can now clone the repo with:

```bash
git clone https://github.com/LAAC-LSCP/VTC.git
cd VTC
```

Finally, you can install python dependencies with the following command:

```bash
uv sync
```

The model weights, config and thresholds are not stored in this repository. They are downloaded automatically from the [`coml/VTC-2`](https://huggingface.co/coml/VTC-2) Hugging Face repository the first time you run inference, and cached locally (in `~/.cache/huggingface/` by default) for later runs.

## 2. Inference

Your audio files should be in the `.wav` format, sampled at 16 kHz and contain a single channel (mono).
If not, you can use the `convert-audios` command to resample your `.wav` files to 16 kHz and average the channels:

```bash
uv run convert-audios --wavs raw_audios --output audios
```

Then run the model on the folder containing your audio files:

```bash
uv run vtc-infer \
    --wavs audios \        # path to the folder containing the audio files
    --output predictions \ # output folder
    --device cpu           # device to run the model on: ('cpu', 'cuda' or 'gpu', 'mps')
```

By default, the latest model (VTC 2.2) and its balanced thresholds are used. Pass `--high-precision` to use the high-precision thresholds instead. The full list of options (batch size, previous model versions, custom thresholds, segment merging, ...) is available with `uv run vtc-infer --help` and in the [documentation](https://laac-lscp.github.io/VTC/guide/#command-line-interface-arguments). VTC 2.0 cannot be run with this version: check out the `v2.0.2` tag instead, as described in the [version history](https://laac-lscp.github.io/VTC/misc/versions/#using-vtc-20).

The model outputs are saved to `<output_folder>/` with the following structure:

```txt
<output_folder>/
│
├── 📂 rttm/          # Final output (with segment merging applied)
├── 📄 rttm.csv       # CSV version of final speaker segments
├── 📄 f1.toml        # Copy of the thresholds used for this run
│
├── 📂 raw_rttm/      # Raw output, without segment merging (only with --keep_raw)
└── 📄 raw_rttm.csv   # CSV version of raw speaker segments (only with --keep_raw)
```

#### Helper script
An example of a bash script is given to perform inference in `run.sh`. Simply set the correct variables in the script and run it:

```bash
sh run.sh
```

#### Python API
Inference can also be run from Python:

```python
from vtc_inference import run_vtc

run_vtc(
    output="predictions",
    wavs="audios",
    config="2.2",
    checkpoint="2.2",
    device="cpu",
)
```

## 3. Model Performance

### 3.1 Runtime
The tables below give the measured processing time as a fraction of the audio duration.

<table>
<tr><th>Table 1: GPU times </th><th>Table 2: CPU times</th></tr>
<tr><td>

| Batch size | Hardware        | Speedup factor |
|:----------:|:----------------|:--------------:|
| 64         | H100            |      1/46      |
| 128        | H100            |     1/117      |
| 256        | H100            |   **1/226**    |

</td><td>

| Batch size | Hardware                      | Speedup factor |
|:----------:|:------------------------------|:--------------:|
| 64         | AMD EPYC 9334 32-Core         |      1/6       |
| 128        | AMD EPYC 9334 32-Core         |      1/7       |
| 256        | AMD EPYC 9334 32-Core         |    **1/7**     |

</td></tr> </table>

On an H100 GPU with a batch size of 256, processing takes about $1/226$ of the audio duration:
- $\approx 16$ seconds for a $1\text{ h}$ recording ($3600 / 226$),
- $\approx 4$ minutes for a $16\text{ h}$ recording ($16 * 3600 / 226$).


On an AMD EPYC 9334 32-Core CPU with a batch size of 256, it takes about $1/7$ of the audio duration:
- $\approx 8.5$ minutes for a $1\text{ h}$ recording ($3600 / 7$),
- $\approx 2.3$ hours for a $16\text{ h}$ recording ($16 * 3600 / 7$).

### 3.2 Model Performance on the heldout set

We evaluate VTC 2.2 on a heldout set and compare it with previous versions and with a second human annotator (Human 2).

| Model   |   KCHI   |   OCH    |   MAL    |   FEM    | Average F1-score |
|---------|:--------:|:--------:|:--------:|:--------:|:----------------:|
| VTC 1.0 |   68.2   |   30.5   |   41.2   |   63.7   |       50.9       |
| VTC 1.5 |   68.4   |   20.6   |   56.7   |   68.9   |       53.6       |
| VTC 2.0 | **71.8** |   51.4   |   60.3   |   74.8   |       64.6       |
| VTC 2.1 |   67.1   |   56.1   |   68.8   |   75.5   |       66.9       |
| VTC 2.2 |   68.5   | **58.1** | **70.9** | **75.6** |     **68.3**     |
| Human 2 |   79.7   |   60.4   |   67.6   |   71.5   |       69.8       |

**Table 1**: F1-scores (%) obtained on the standard test set by VTC 1.0, VTC 1.5, VTC 2.0, VTC 2.1, VTC 2.2, and a second human annotator.
The best model is indicated in bold.

VTC 2.2 has the highest average F1-score of all versions, close to the agreement between two human annotators. VTC 2.0 remains best on KCHI.

---
## 4. Citation
The training code for BabyHuBERT can be found here: [LAAC-LSCP/BabyHuBERT](https://github.com/LAAC-LSCP/BabyHuBERT)

To cite this work, please use the following bibtex.

```bibtex
@inproceedings{charlot26_interspeech,
  title     = {{BabyHuBERT: Multilingual Self-Supervised Learning for Segmenting Speakers in Child-Centered Long-Form Recordings}},
  author    = {Théo Charlot and Tarek Kunze and Maxime Poli and Alejandrina Cristia and Emmanuel Dupoux and Marvin Lavechin},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
  pages     = {4896--4901},
  doi       = {10.21437/Interspeech.2026-2772},
  issn      = {2958-1796},
}
```

Or the following APA formatted citation
```APA
Charlot, T., Kunze, T., Poli, M., Cristia, A., Dupoux, E., Lavechin, M. (2026) BabyHuBERT: Multilingual Self-Supervised Learning for Segmenting Speakers in Child-Centered Long-Form Recordings. Proc. Interspeech 2026, 4896-4901, doi: 10.21437/Interspeech.2026-2772
```

## 5. Acknowledgement
The following publications document earlier versions of the Voice Type Classifier, in reverse chronological order.

### VTC 1.5 (Whisper-VTC)
GitHub repository: [github.com/LAAC-LSCP/VTC-IS-25](https://github.com/LAAC-LSCP/VTC-IS-25)
```bibtex
@inproceedings{kunze25_interspeech,
    title     = {{Challenges in Automated Processing of Speech from Child Wearables:  The Case of Voice Type Classifier}},
    author    = {Tarek Kunze and Marianne Métais and Hadrien Titeux and Lucas Elbert and Joseph Coffey and Emmanuel Dupoux and Alejandrina Cristia and Marvin Lavechin},
    year      = {2025},
    booktitle = {{Interspeech 2025}},
    pages     = {2845--2849},
    doi       = {10.21437/Interspeech.2025-1962},
    issn      = {2958-1796},
}
```

### VTC 1.0 (PyanNet-VTC)
GitHub repository: [github.com/MarvinLvn/voice-type-classifier](https://github.com/MarvinLvn/voice-type-classifier)
```bibtex
@inproceedings{lavechin20_interspeech,
    title     = {An Open-Source Voice Type Classifier for Child-Centered Daylong Recordings},
    author    = {Marvin Lavechin and Ruben Bousbib and Hervé Bredin and Emmanuel Dupoux and Alejandrina Cristia},
    year      = {2020},
    booktitle = {Interspeech 2020},
    pages     = {3072--3076},
    doi       = {10.21437/Interspeech.2020-1690},
    issn      = {2958-1796},
}
```

This work uses the [segma](https://github.com/arxaqapi/segma) library which is heavily inspired by [pyannote.audio](https://github.com/pyannote/pyannote-audio).

This work was performed using HPC resources from GENCI-IDRIS (Grant 2024-AD011015450 and 2025-AD011016414) and was developed as part of the ExELang project funded by the European Union (ERC, ExELang, Grant No 101001095).
