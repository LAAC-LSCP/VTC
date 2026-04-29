---
icon: lucide/file-sliders
---

# User Guide
Make sure you've completed each step outlined on the [Getting Started](getting-started.md) page before continuing.


## Run VTC

To be able to run VTC, place your `.wav` files in a folder `audio_folder`  and run the following command in a terminal while pointing the script to the audio folder path:

```bash
uv run scripts/infer.py      \
    --wavs <audio_folder>    \
    --output <output_folder> \
    --device cpu
```

A helper script is also provided at `scripts/run.sh`. Before using it, open the file in a text editor and change the following variables at the top of the script:

- `WAVS` — set this to the path of the folder containing your `.wav` files (e.g., `WAVS="/home/user/my_project/audio"`)
- `OUTPUT` — set this to the path where you want the results saved (e.g., `OUTPUT="/home/user/my_project/output"`)
- `DEVICE` — set this to the hardware you want to use: `cpu`, `cuda` (for NVIDIA GPUs), or `mps` (for Apple Silicon Macs)

Then run the script:

```bash
sh scripts/run.sh
```

For more arguments check the [Command Line Interface Arguments](#command-line-interface-arguments) section.


## Understanding outputs

After running VTC, you get the following structure on disk with the `📂 rttm/` folder containing one RTTM file per audio. 

```
<output_folder>/
├── 📂 rttm/          # Final segments (one .rttm file per audio)
└── 📄 rttm.csv       # Final segments as a single CSV
```

VTC produces results in two formats: a single **CSV file** and individual **RTTM files**. Both contain the same information — use whichever is more convenient for your workflow.

### RTTM format

The `📂 rttm/` folder contains one RTTM file per audio file. RTTM is a standard format in speech processing. Each line represents one detected speech segment:

```
SPEAKER <uid> 1 <start_time_s> <duration_s> <NA> <NA> <label> <NA> <NA>
```
Only four fields are relevant: `uid` (the filename), `start_time_s` (when the segment starts, in seconds), `duration_s` (how long it lasts, in seconds), and `label` (the speaker type: KCHI, OCH, MAL, or FEM).
The remaining fields are placeholders (`1` and `<NA>`) required by the RTTM format specification but not used by VTC.

### CSV format (recommended)

**Use `rttm.csv` for analysis.** 

You can open it in any spreadsheet application or preferably load it as a dataframe using R or Python. Each row is one speech segment with the same four relevant fields as the RTTM format: a filename (`uid`), start time (`start_time_s`), duration (`duration_s`), and the assigned speaker label (`label`).

Here is an example of a few detected segments, followed by an illustration of what those segments look like when laid out on a timeline. This visualization is provided here for explanatory purposes only — VTC does not generate this image. You will only receive the CSV and RTTM files described above.

```csv
uid,              start_time_s, duration_s, label
recording_jd7aks,         0.12,       1.20,   FEM
recording_jd7aks,         3.30,       0.34,   MAL
recording_jd7aks,         3.98,       1.98,   FEM
recording_jd7aks,         5.86,       2.10,   MAL
recording_jd7aks,         6.10,       1.90,  KCHI
recording_jd7aks,         8.24,       0.52,   OCH
```
![Detected speech segments layed-out in the time dimension](assets/recording_jd7aks.png)


## Speed

| Setup | Speedup | 1h audio | 16h audio |
|-------|---------|----------|-----------|
| H100 GPU, batch 256 | 1/905 | ~4 sec | ~1 min |
| A40 GPU, batch 256 | 1/650 | ~6 sec | ~1.5 min |
| CPU (Xeon Silver), batch 64 | 1/16 | ~4 min | ~1 hour |

GPU processing is strongly recommended for large corpora. If you get out-of-memory errors, reduce the batch size.

To change the batch size, add the `--batch_size` argument to your command. The default is 128. For example, to use a batch size of 64:

```bash
uv run scripts/infer.py      \
    --wavs <audio_folder>    \
    --output <output_folder> \
    --device cuda             \
    --batch_size 64
```

Larger batch sizes are faster but require more memory. If VTC crashes with a "CUDA out of memory" error, try lowering the batch size (e.g., 64 or 32) until it runs successfully.


## Command Line Interface Arguments
Here is the complete list of arguments you can use when running VTC.

| <div style="width: 140px;">Argument</div> | Default    | Description                                                        |
|-----------------------|--------------------------------|--------------------------------------------------------------------|
| `--config`            | `VTC-2.0/model/config.yml`     | Config file to be loaded and used for inference.                   |
| `--checkpoint`        | `VTC-2.0/model/best.ckpt`      | Path to a pretrained model checkpoint.                             |
| `--wavs`              | **required**.                  | Folder containing the audio files to run inference on.             |
| `--output`            | **required**                   | Output path to the folder that will contain the final predictions. |
| `--uris`              | —                              | Path to a file containing the list of URIs to use.                 |
| `--save_logits`       | `False`                        | Save the logits to disk. Can be memory intensive.                  |
| `--thresholds`        | —                              | Path to a thresholds dict to perform predictions via thresholding. |
| `--min_duration_on_s` | `0.1`                          | Remove speech segments shorter than that many seconds.             |
| `--min_duration_off_s`| `0.1`                          | Fill same-speaker gaps shorter than that many seconds.             |
| `--batch_size`        | `128`                          | Batch size for the forward pass of the model.                      |
| `--recursive_search`  | `False`                        | Recursively search for `.wav` files. May be slow.                  |
| `--device`            | `cuda`                         | Device to use. Choices: `gpu`, `cuda`, `cpu`, `mps`.               |
| `--keep_raw`          | `False`                        | Keep raw RTTM and save to `<output>/raw_rttm/`.                    |

### Example commands

**Basic usage on CPU** — process all `.wav` files in `my_audio/` and save results to `my_output/`:

```bash
uv run scripts/infer.py --wavs my_audio/ --output my_output/ --device cpu
```

**GPU with a smaller batch size** — useful if your GPU has limited memory:

```bash
uv run scripts/infer.py --wavs my_audio/ --output my_output/ --device cuda --batch_size 64
```

**Search subfolders for audio files** — if your `.wav` files are organized in subdirectories (e.g., one per participant):

```bash
uv run scripts/infer.py --wavs my_audio/ --output my_output/ --device cuda --recursive_search
```

**Keep raw (unprocessed) output** — save the pre-post-processing RTTM files for debugging or custom analysis:

```bash
uv run scripts/infer.py --wavs my_audio/ --output my_output/ --device cuda --keep_raw
```

**Filter out very short segments** — ignore detected segments shorter than 0.2 seconds and fill same-speaker gaps shorter than 0.15 seconds:

```bash
uv run scripts/infer.py --wavs my_audio/ --output my_output/ --device cuda \
    --min_duration_on_s 0.2 --min_duration_off_s 0.15
```


<!-- ## test

<iframe
  src="/assets/visualisations/audio-annotations-player.html"
  width="100%" height="760"
  style="border:1px solid var(--md-default-fg-color--lightest);border-radius:8px"
  loading="lazy"></iframe> -->
