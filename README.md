# Voice Type Classifier (VTC)


## Usage
Ensure that you have uv installed on you system.

Clone the repo and setup the dependencies:

```
git clone https://github.com/LAAC-LSCP/VTC.git
cd VTC

uv sync
```
The audio files for inference simply need to lie in a simple repository, the inference script will load them automatically.

### Inference

Inference is done using a checkpoint of the model, linking the corresponding config file used for training and the list of audio files to run the inference on. For more fine-grained tuning, use the `min-duration-on-s` and `min-duration-off-s` parameters.

```bash
uv run scripts/infer.py \
    --wavs audios \
    --output predictions
```

The output of the model will be in `<output_folder>/rttm`. The raw outputs without segment merging applied are present in `<output_folder>/raw_rttm`.

### Helper script

To perform inference and speech segment merging (see merge_segments.py for help or [this pyannote.audio description](https://github.com/pyannote/pyannote-audio/blob/240a7f3ef60bc613169df860b536b10e338dbf3c/pyannote/audio/pipelines/resegmentation.py#L79-L82)), a single bash script is given.

Simply set the correct variables in the script and run it:

```bash
sh scripts/run.sh
````

---
## Citation
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

## Acknowledgement
The Voice Type Classifier has benefited from numerous contributions over time, following publications document its evolution, listed in reverse chronological order.

#### 1. Whisper-**VTC**
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

#### 2. PyanNet-VTC (Original)
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