---
icon: lucide/package-open
# icon: lucide/play
---

# Getting Started

## Requirements

- **OS**: Linux or macOS (Windows is not supported)
- **System tools**: [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org/), [git-lfs](https://git-lfs.com/)
<!-- - **Python**: 3.13+ -->

## Installation

Install the required system dependencies, then clone and set up VTC:

!!! info "Installing python"
    The correct python versions and its packages are automatically handled by uv, no need to install python yourself.

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

=== "MacOS"

    ```bash
    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Install brew (brew is a  package manager for MacOS)
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Install  ffmpeg and git-lfs (MacOS)
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
    VTC is included as a git submodule, and the model weights live inside it.
    Without `--recurse-submodules`, the submodule stays empty, the weights are never fetched, and VTC will fail to load at runtime.
    If you've already cloned without the flag, run:
    ```bash
        git submodule update --init --recursive
    ```


## Prepare your audio

### Recommended folder structure

Before running VTC, it helps to organize your files in a clear structure. Here is a recommended layout:

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

Place all of your `.wav` files inside a single folder (e.g., `audio/`). When you run VTC, you will point it to this folder, and VTC will process every `.wav` file it finds inside. The results will be written to a separate output folder that VTC creates for you.

!!! tip "Subfolders"
    If your audio files are organized in subfolders (e.g., one subfolder per child), you can use the `--recursive_search` flag to tell VTC to look inside subfolders. See the [Command Line Interface Arguments](guide.md#command-line-interface-arguments) for details.

### Audio format requirements

VTC expects **WAV** files sampled at **16 kHz** and with a single channel (mono). You can check your files with:

```bash
ffprobe your_recording.wav
```
When you run the command above, `ffprobe` will print information about your file. Look for a line that starts with `Stream #0:0: Audio:` — this is where the important details appear. You should see something like:
```
Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, 1 channels, s16, 256 kb/s
```
Here's what to check:

- **`pcm_s16le`**: confirms the file is an uncompressed WAV.
- **`16000 Hz`**: the sample rate. It must be exactly 16 kHz (i.e. `16000 Hz`).
- **`mono`** (or **`1 channels`**): the file has a single audio channel. Depending on your `ffprobe` version, this may appear as the word `mono` or as `1 channels`. If you see `stereo`, `2 channels`, or any number higher than 1, the file has multiple channels and will need to be converted.

If any of these values don't match, you'll need to convert your file before running VTC.


In that case, use the included script or ffmpeg directly. Both will resample to 16 kHz and average across channels to produce a single mono file:

<!-- ```bash
# Using the provided script
uv run scripts/convert.py --input /path/to/raw_audio --output /path/to/converted

# Or manually with ffmpeg (works with MP3, FLAC, M4A, etc.)
ffmpeg -i input.mp3 -acodec pcm_s16le -ar 16000 -ac 1 output.wav
``` -->

=== "FFmpeg"

    ```bash
    # Replace `raw_audio.mp3` with the name of your file (it can be `.mp3`, `.flac`, etc.),
    # and `converted.wav` with whatever you'd like to name the output file.

    ffmpeg -i raw_audio.mp3 -acodec pcm_s16le -ar 16000 -ac 1 converted.wav
    ```

=== "Python conversion script"

    ```bash
    # Replace `/path/to/raw_audio` with the folder containing the audio files,
    # and `/path/to/converted` with the name of the output folder

    uv run scripts/convert.py --input /path/to/raw_audio --output /path/to/converted
    ```



## Common errors

If you run into problems during installation or when running VTC, check the table below for quick fixes. A more detailed list is available on the [Troubleshooting](misc/troubleshooting.md) page.

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `uv: command not found` | uv is not installed | For detailed installation instructions, see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) |
| `ffmpeg: command not found` | ffmpeg is not installed | `sudo apt install ffmpeg` (Linux) or `brew install ffmpeg` (macOS) |
| Model weights missing | Cloned without `--recurse-submodules` | Run `git lfs install && git submodule update --init --recursive` |
| `CUDA out of memory` | Batch size is too large for your GPU | Add `--batch_size 64` (or lower) to your command, or use `--device cpu` |
| No `.wav` files found | Wrong folder or wrong audio format | Make sure the `--wavs` path points to a folder containing `.wav` files |
