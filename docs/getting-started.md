---
icon: lucide/package-open
---

# Getting Started

This page covers installing VTC and preparing your audio files. If something goes wrong, see [Common errors](#common-errors) at the bottom of this page or the [Troubleshooting](misc/troubleshooting.md) page.

---

## Requirements

- **Operating system:** Linux or macOS. Windows is not supported. If you only have a Windows machine, the easiest workaround is to use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install).
- **[`uv`](https://docs.astral.sh/uv/)**, which installs Python and the necessary libraries.
- **[`ffmpeg`](https://ffmpeg.org/) version 4 to 8**, which VTC uses to read audio files. FFmpeg 9 is not supported yet.
- **An internet connection on the first run**, to download the model (see [Where the model comes from](#where-the-model-comes-from)).
<!-- - **Python**: 3.13+ -->

!!! info "You don't need to install Python yourself"
    `uv` installs the right Python version and all required packages.

---

## Installation

Run the block for your operating system in a terminal.

=== "Linux (Ubuntu/Debian)"

    ```bash
    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Install ffmpeg (Ubuntu/Debian)
    sudo apt install ffmpeg

    # Clone the repo
    git clone https://github.com/LAAC-LSCP/VTC.git
    cd VTC

    # Install Python dependencies
    uv sync

    # Verify everything is set up
    ./check_sys_dependencies.sh
    ```

=== "macOS"

    ```bash
    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Install brew (a package manager for macOS)
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Install ffmpeg 8 (Homebrew's default ffmpeg is version 9, which VTC cannot use)
    brew install ffmpeg@8
    echo 'export PATH="$(brew --prefix ffmpeg@8)/bin:$PATH"' >> ~/.zshrc
    echo 'export DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix ffmpeg@8)/lib"' >> ~/.zshrc
    source ~/.zshrc

    # Clone the repo
    git clone https://github.com/LAAC-LSCP/VTC.git
    cd VTC

    # Install Python dependencies
    uv sync

    # Verify everything is set up
    ./check_sys_dependencies.sh
    ```

### Where the model comes from

The model is not stored in the repository. On the first run, VTC downloads the model weights, configuration and [threshold](advanced/thresholds.md) files from the [`coml/VTC-2`](https://huggingface.co/coml/VTC-2) repository on Hugging Face. Later runs use the downloaded copy.

??? info "Where the downloaded model is stored"
    The model is stored in the Hugging Face cache, `~/.cache/huggingface/hub/` by default. To use another location, for example a larger disk, set the `HF_HOME` environment variable before running VTC:
    ```bash
    export HF_HOME=/path/with/more/space/huggingface
    ```

??? note "Upgrading from an older install"
    VTC 2.1 and earlier needed `git-lfs` and `git clone --recurse-submodules` to fetch the model into a `VTC-2/` folder. Neither is needed anymore, and you can delete the `VTC-2/` folder. See [Version History](misc/versions.md#migrating-from-vtc-20-or-21) for the other changes.

---

## Prepare your audio

### Folder structure

Put all the recordings you want to process in one folder. VTC processes every `.wav` file in that folder and writes its results to a separate output folder:

```
my_project/
├── audio/               # Your WAV files go here
│   ├── child01_day1.wav
│   ├── child01_day2.wav
│   └── child02_day1.wav
└── output/              # VTC will write results here
    ├── rttm/
    │   ├── child01_day1.rttm
    │   └── ...
    └── rttm.csv
```

!!! tip "Recordings in subfolders"
    If your recordings are sorted into subfolders (for example one per child), add the `--recursive_search` flag. See [Command Line Interface Arguments](guide.md#command-line-interface-arguments).

### Audio format

Each recording must be:

- a **WAV file** (uncompressed audio),
- sampled at **16 kHz** (16 000 samples per second),
- **mono** (a single channel).

To check a file, run `ffprobe` (installed with `ffmpeg`):

```bash
ffprobe your_recording.wav
```

Look for the line starting with `Stream #0:0: Audio:`:

```
Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, 1 channels, s16, 256 kb/s
```

- `pcm_s16le` means the file is uncompressed WAV.
- `16000 Hz` is the sample rate. It must be exactly 16 kHz.
- `1 channels` (or `mono`, depending on your `ffprobe` version) means the file has a single channel. `stereo` or `2 channels` means it needs converting.

!!! warning "VTC does not check the sample rate"
    A file at another sample rate, such as 44.1 or 48 kHz, is processed without any error or warning, but its results are wrong. Check your files before running VTC. Files with more than one channel are skipped with an error.

If any of these don't match, convert the file with one of the methods below. Both resample to 16 kHz and mix all channels into one. For other formats such as `.mp3` or `.flac`, use FFmpeg: `convert-audios` only converts `.wav` files and ignores all others.

=== "FFmpeg (one file at a time)"

    ```bash
    # Replace `raw_audio.mp3` with the name of your file (it can be `.mp3`, `.flac`, etc.),
    # and `converted.wav` with whatever you'd like to name the output file.

    ffmpeg -i raw_audio.mp3 -acodec pcm_s16le -ar 16000 -ac 1 converted.wav
    ```

=== "convert-audios (a whole folder of WAV files)"

    ```bash
    # Replace `/path/to/raw_audio` with the folder containing the .wav files,
    # and `/path/to/converted` with the name of the output folder

    uv run convert-audios --wavs /path/to/raw_audio --output /path/to/converted
    ```

    `convert-audios` also searches subfolders of `/path/to/raw_audio`, and writes every converted file directly into `/path/to/converted` under its original name. It refuses to upsample files recorded below 16 kHz unless you add `--allow_upsampling`, because upsampling cannot recover the missing audio detail.

---

## Common errors

See [Troubleshooting](misc/troubleshooting.md) for problems not listed here.

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `uv: command not found` | uv is not installed | See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) |
| `ffmpeg: command not found` | ffmpeg is not installed | `sudo apt install ffmpeg` (Linux) or see the [macOS installation](#installation) steps |
| `RuntimeError: Could not load libtorchcodec` | FFmpeg is missing or too recent (version 9 or later) | Install FFmpeg 4 to 8; on macOS, follow the [installation](#installation) steps for `ffmpeg@8` |
| Model download fails on first run | No internet access (common on compute-cluster nodes) | Run VTC once on a machine with internet access to fill the cache; see [HPC / SLURM setup](advanced/hpc.md#offline-compute-nodes) |
| `CUDA out of memory` | Batch size too large for your GPU | Add `--batch_size 64` (or lower) to your command, or use `--device cpu` |
| `FileNotFoundError: ... raw_rttm` | No file could be processed: the folder contains no `.wav` files, or all of them failed | Check that the path passed to `--wavs` contains 16 kHz mono `.wav` files (use `--recursive_search` for subfolders) |
