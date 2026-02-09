# Voice Type Classifier (VTC) 2.0

The Voice Type Classifier is a classification model that given a input audio file, outputs a precise segmentation of speakers.

The four classes that the model will output are:
- **FEM** stands for adult female speech
- **MAL** stands for adult male speech
- **KCHI** stands for key-child speech
- **OCH** stands for other child speech

The model has been specifically trained to work with child-centered long-form recordings. These are recordings that can span multiple hours and have been collected using a portable recorder attached to the vest of a child (usually 0 to 5 years of age).

## 0. Table of content

1. [Installation](#1-installation)
2. [Inference](#2-inference)
3. [Model Performance](#3-model-performance)
4. [Citation](#4-citation)
5. [Acknowledgement](#5-acknowledgement)

## 1. Installation

To use the model, you will need a unix-based machine (Linux or MacOS) and python version 3.13 or higher installed. Windows is not supported for the moment.
As system dependencies, ensure that [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org/), and [git-lfs](https://git-lfs.com/) are installed. You can check that by running:

```bash
./check_sys_dependencies.sh
```

You can now clone the repo with:

```bash
git lfs install
git clone --recurse-submodules https://github.com/LAAC-LSCP/VTC.git
cd VTC
```

Finally, you can install python dependencies with the following command:

```bash
uv sync
```


## 2. Inference

Inference is done using a checkpoint of the model, linking the corresponding config file used for training and the list of audio files to run the model on. You audio files should be in the `.wav` format, sampled at 16 000 kHz and contain a single channel (mono).
If not, you can use the `scripts/convert.py` file to convert your audios to 16 000 Hz and average the channels.


```bash
uv run scripts/infer.py \
    --wavs audios \        # path to the folder containing the audio files
    --output predictions \ # output folder
    --device cpu           # device to run the model on: ('cpu', 'cuda' or 'gpu', 'mps')
```

The model outputs are saved to `<output_folder>/` with the following structure:

```txt
<output_folder>/
│
├── 📂 rttm/          # Final output (with segment merging applied)
├── 📂 raw_rttm/      # Raw output (without segment merging)
├── 📄 rttm.csv       # CSV version of final speaker segments
└── 📄 raw_rttm.csv   # CSV version of raw speaker segments
```

> [!NOTE]
> Segment merging is applied to the main output. See the [pyannote.audio description](https://github.com/pyannote/pyannote-audio/blob/240a7f3ef60bc613169df860b536b10e338dbf3c/pyannote/audio/pipelines/resegmentation.py#L79-L82) for details.

#### Helper script
An example of a bash script is given to perform inference in `scripts/run.sh`. Simply set the correct variables in the script and run it:

```bash
sh scripts/run.sh
````

## 3. Model Performance

### 3.1 Runtime
We tested the inference pipeline on multiple GPUs and CPUs and display the expected speedup factors that can be used to estimate the total duration needed to process $x$ hours of audio.

<table>
<tr><th>Table 1: GPU times </th><th>Table 2: CPU times</th></tr>
<tr><td>

| Batch size | Hardware        | Speedup factor |
|:----------:|:----------------|:--------------:|
| 64         | Quadro RTX 8000 | 1/152          |
| 128        | Quadro RTX 8000 | 1/286          |
| 256        | Quadro RTX 8000 | 1/531          |
| 64         | A40             | 1/450          |
| 128        | A40             | 1/358          |
| 256        | A40             | 1/650          |
| 64         | H100            | 1/182          |
| 128        | H100            | 1/466          |
| 256        | H100            | **1/905**      |

</td><td>

| Batch size | Hardware                      | Speedup factor|
|:----------:|:------------------------------|:-------------:|
| 64         | Intel(R) Xeon(R) Silver 4214R | 1/16          |
| 128        | Intel(R) Xeon(R) Silver 4214R | 1/15          |
| 256        | Intel(R) Xeon(R) Silver 4214R | 1/16          |
| 64         | AMD EPYC 7453 28-Core         | 1/20          |
| 128        | AMD EPYC 7453 28-Core         | 1/21          |
| 256        | AMD EPYC 7453 28-Core         | 1/22          |
| 64         | AMD EPYC 9334 32-Core         | 1/25          |
| 128        | AMD EPYC 9334 32-Core         | 1/26          |
| 256        | AMD EPYC 9334 32-Core         | **1/29**      |

</td></tr> </table>

<!-- [297353] seconds in 328.475611 s -->
It takes approximatively $1/905$ of the audio duration to run the model with a batch size of 256 on an H100 GPU.
- For a $1\text{ h}$ long audio, the inference will run for approximatively $\approx 4$ seconds. ($3600 / 905$)
- For a $16\text{ h}$ longform audio, the inference will run for $\approx 1 \text{ minute}$ and $4 \text{ seconds}$. ($16 * 3600 / 905$)


On a Intel(R) Xeon(R) Silver 4214R CPU with a batch size of 64, the inference pipeline will be quite slow:
- For a $1\text{ h}$ long audio, the inference will run for approximatively $\approx 4$ minutes. ($3600 / 15$)
- For a $16\text{ h}$ longform audio, the inference will run for $\approx 1 \text{ hour}$ and $4 \text{ minutes}$. ($16 * 3600 / 15$)

### 3.2 Model Performance on the heldout set

We evaluate the new model, VTC 2.0, on a heldout set and compare it to the previous models and the Human performance (Human 2).

| Model          | KCHI |  OCH |  MAL |  FEM | Average F1-score |
|----------------|:----:|:----:|:----:|:----:|:--------:|
| VTC 1.0        | 68.2 | 30.5 | 41.2 | 63.7 |   50.9   |
| VTC 1.5        | 68.4 | 20.6 | 56.7 | 68.9 |   53.6   |
| VTC 2.0        | **71.8** | **51.4** | **60.3** | **74.8** | **64.6** |
| Human 2        | 79.7 | 60.4 | 67.6 | 71.5 |   69.8   |

**Table 1**: F1-scores (%) obtained on the standard test set VTC 1.0, VTC 1.5, VTC 2.0, and a second human annotator.
The best model is indicated in bold.

As displayed in table 1, our model performs better than previous iterations with performances close to the Human performances. VTC 2.0 even surpasses human like performance on the **FEM** class.

### 3.3 Confusion Matrices on the heldout set
- **OVL**: is the overlap between speakers.
- **SIL**: are the section with silence/noise.

<p float="left" align="middle">
  <img src="figures/vtc2_heldout_full_cm_precision.png" width="400"/>
  <img src="figures/vtc2_heldout_full_cm_recall.png" width="400"/> 
</p>

---
## 4. Citation
The training code for BabyHuBERT can be found here: [LAAC-LSCP/BabyHuBERT](https://github.com/LAAC-LSCP/BabyHuBERT)

To cite this work, please use the following bibtex.

```bibtex
@misc{charlot2025babyhubertmultilingualselfsupervisedlearning,
    title={BabyHuBERT: Multilingual Self-Supervised Learning for Segmenting Speakers in Child-Centered Long-Form Recordings}, 
    author={Théo Charlot and Tarek Kunze and Maxime Poli and Alejandrina Cristia and Emmanuel Dupoux and Marvin Lavechin},
    year={2025},
    eprint={2509.15001},
    archivePrefix={arXiv},
    primaryClass={eess.AS},
    url={https://arxiv.org/abs/2509.15001}, 
}
```

## 5. Acknowledgement
The Voice Type Classifier has benefited from numerous contributions over time, following publications document its evolution, listed in reverse chronological order.

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

This work uses the [segma](https://github.com/arxaqapi/segma) library which is heavely inspired by [pyannote.audio](https://github.com/pyannote/pyannote-audio).

This work was performed using HPC resources from GENCI-IDRIS (Grant 2024-AD011015450 and 2025-AD011016414) and was developed as part of the ExELang project funded by the European Union (ERC, ExELang, Grant No 101001095).
