# VTC 2.0 Evaluation

## Runtime
The model has been tested on a Nvidia H100 NVL with 94 GB of VRAM and an AMD EPYC 9334 CPU.

| Batch size | GPU   | CPU   |
|:----------:|:-----:|:-----:|
| 128        | 1/13  | -     |


<!-- 297353 seconds in 22612.4552 s -->
It takes approximatively $1/13$ of the audio duration to run the model.
- For a $1\text{ h}$ long audio, the inference will run for approximatively $\approx 5$ minutes.
- For a $16\text{ h}$ longform audio, the model will run for $\approx 1 \text{ hour and } 14 \text{ minutes}$.




## Model Performance on the heldout set

We evaluate the new model, BabyHuBERT-VTC, on a heldout set and compare it to the previous models and the Human performance (Human 2).

| Model          | KCHI |  OCH |  MAL |  FEM | F1-score |
|----------------|:----:|:----:|:----:|:----:|:--------:|
| Human 2        | 79.7 | 60.4 | 67.6 | 71.5 |   69.8   |
| PyanNet-VTC    | 68.2 | 30.5 | 41.2 | 63.7 |   50.9   |
| Whisper-VTC    | 68.4 | 20.6 | 56.7 | 68.9 |   53.6   |
| BabyHuBERT-VTC | 71.8 | 51.4 | 60.3 | 74.8 |   64.6   |

**Table 1**: F1-scores (%) obtained on the standard test set by
PyanNet-VTC, Whisper-VTC, a second human annotator (Human 2) and the **best BabyHuBERT-VTC** model.

As displayed in table 1, our model performs better than previous iterations with performances close to the Human performances. BabyHuBERT-VTC even surpasses human like performance on the **FEM** class.

## Confusion Matrices on the heldout set
- **OVL**: is the overlap between speakers.
- **SIL**: are the section with silence/noise.

<p float="left" align="middle">
  <img src="figures/bbh_heldout_full_cm_precision.png" width="400"/>
  <img src="figures/bbh_heldout_full_cm_recall.png" width="400"/> 
</p>
