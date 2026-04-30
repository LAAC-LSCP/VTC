---
icon: lucide/package-open
---

# Getting Started

> **This page walks you through installing VTC and preparing your audio files. By the end, your machine will be ready to run VTC and your recordings will be in the format VTC expects.**

If anything goes wrong, see the [Common errors](#common-errors) table at the bottom of this page, or the more detailed [Troubleshooting](misc/troubleshooting.md) page.

---

## Requirements

> **What you need before installing VTC: a supported operating system and three small command-line tools.**

- **Operating system:** Linux or macOS. *Windows is not supported.* If you only have a Windows machine, the easiest workaround is to use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install).
- **Three small tools** that VTC depends on:
    - [`uv`](https://docs.astral.sh/uv/) — manages the right version of Python and the libraries VTC needs, so you don't have to.
    - [`ffmpeg`](https://ffmpeg.org/) — converts audio between formats.
    - [`git-lfs`](https://git-lfs.com/) — fetches the trained model files (which are too large for ordinary `git`).
<!-- - **Python**: 3.13+ -->

!!! info "You don't need to install Python yourself"
    `uv` handles the correct Python version and all of its packages automatically. Just install `uv` (instructions below) and you're set.

---

## Installation

> **Copy and paste the block matching your operating system into a terminal. The whole process is about five commands.**

=== "Linux (Ubuntu/Debian)"

    ```bash
    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Install ffmpeg and git-lfs (Ubuntu/Debian)
    sudo apt install ffmpeg git-lfs

    # Clone the repo (--recurse-submodules is required for model weights)
    git lfs install
    git clone --recurse-submodules https://github.com/LAAC-LSCP/VTC.git
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

    # Install ffmpeg and git-lfs (macOS)
    brew install ffmpeg git-lfs

    # Clone the repo (--recurse-submodules is required for model weights)
    git lfs install
    git clone --recurse-submodules https://github.com/LAAC-LSCP/VTC.git
    cd VTC

    # Install Python dependencies
    uv sync

    # Verify everything is set up
    ./check_sys_dependencies.sh
    ```

??? warning "Don't skip `--recurse-submodules`"
    The trained VTC model is stored as a separate piece (a "submodule") inside the repository. Without `--recurse-submodules`, that piece stays empty, the model weights never get downloaded, and VTC will fail when you try to run it.

    If you've already cloned without that flag, you can fix it after the fact:
    ```bash
    git submodule update --init --recursive
    ```

---

## Prepare your audio

> **VTC only reads WAV files at 16 kHz, mono. This section shows you how to organise your files and how to convert them if they're in a different format.**

### Recommended folder structure

Before running VTC, organise your files so that all the recordings you want to process live together in one folder. A simple layout:

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

Place all of your `.wav` files inside a single folder (e.g., `audio/`). When you run VTC, you point it at this folder, and VTC processes every `.wav` file it finds inside. Results are written to a separate folder that VTC creates for you.

!!! tip "If your audio is organised in subfolders"
    If your recordings are sorted into subfolders (e.g., one subfolder per child), use the `--recursive_search` flag to tell VTC to look inside subfolders too. See the [Command Line Interface Arguments](guide.md#command-line-interface-arguments) for details.

### Audio format requirements

VTC expects each recording to be:

- a **WAV file** (uncompressed audio),
- sampled at **16 kHz** (16 000 samples per second), and
- with a **single audio channel** (mono — not stereo).

You can check a file's format with the `ffprobe` tool (it comes with `ffmpeg`):

```bash
ffprobe your_recording.wav
```

`ffprobe` prints a lot of information; the line you want starts with `Stream #0:0: Audio:` and looks something like this:

```
Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, 1 channels, s16, 256 kb/s
```

What to look for on that line:

- **`pcm_s16le`** — confirms the file is an uncompressed WAV.
- **`16000 Hz`** — the sample rate. It must be exactly 16 kHz (`16000 Hz`).
- **`mono`** or **`1 channels`** — the file has a single channel. (Depending on your `ffprobe` version, this appears either as the word `mono` or as `1 channels`. If you see `stereo`, `2 channels`, or anything higher than 1, the file has multiple channels and needs converting.)

If any of these don't match, convert your file before running VTC. Either of the two methods below will resample to 16 kHz and combine multiple channels into a single mono channel:

<!-- ```bash
# Using the provided script
uv run scripts/convert.py --input /path/to/raw_audio --output /path/to/converted

# Or manually with ffmpeg (works with MP3, FLAC, M4A, etc.)
ffmpeg -i input.mp3 -acodec pcm_s16le -ar 16000 -ac 1 output.wav
``` -->

=== "FFmpeg (one file at a time)"

    ```bash
    # Replace `raw_audio.mp3` with the name of your file (it can be `.mp3`, `.flac`, etc.),
    # and `converted.wav` with whatever you'd like to name the output file.

    ffmpeg -i raw_audio.mp3 -acodec pcm_s16le -ar 16000 -ac 1 converted.wav
    ```

=== "Python conversion script (a whole folder at once)"

    ```bash
    # Replace `/path/to/raw_audio` with the folder containing the audio files,
    # and `/path/to/converted` with the name of the output folder

    uv run scripts/convert.py --input /path/to/raw_audio --output /path/to/converted
    ```

---

## Common errors

> **Quick fixes for the problems most people run into. If your issue isn't listed here, see the full [Troubleshooting](misc/troubleshooting.md) page.**

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `uv: command not found` | uv is not installed | See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) |
| `ffmpeg: command not found` | ffmpeg is not installed | `sudo apt install ffmpeg` (Linux) or `brew install ffmpeg` (macOS) |
| Model weights missing | Cloned without `--recurse-submodules` | Run `git lfs install && git submodule update --init --recursive` |
| `CUDA out of memory` | Batch size too large for your GPU | Add `--batch_size 64` (or lower) to your command, or use `--device cpu` |
| No `.wav` files found | Wrong folder, or files are in a different format | Check that the path passed to `--wavs` contains `.wav` files |
