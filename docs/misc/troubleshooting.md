---
icon: lucide/bug-off
---

# Troubleshooting

## Installation

| Error                                        | Fix                                                                            |
| -------------------------------------------- | ------------------------------------------------------------------------------ |
| `uv: command not found`                      | `curl -LsSf https://astral.sh/uv/install.sh \| sh`, then restart your terminal |
| `ffmpeg: command not found`                  | `sudo apt install ffmpeg` (Linux) or `brew install ffmpeg@8` (macOS, see [Installation](../getting-started.md#installation)) |
| `RuntimeError: Could not load libtorchcodec` | FFmpeg is missing or too recent. VTC needs FFmpeg 4 to 8; Homebrew's default `ffmpeg` is version 9. On macOS, follow the `ffmpeg@8` steps in [Installation](../getting-started.md#installation) |
| `uv sync` fails                              | Run it from inside the `VTC/` folder. uv downloads Python 3.13 itself; if that download fails, run `uv python install 3.13` and check the error message |
| `vtc-infer` not found / `No such file or directory: scripts/infer.py` | Pull the latest version and run `uv sync`; the command is now `uv run vtc-infer` (see [Version History](versions.md#migrating-from-vtc-20-or-21)) |

## Running VTC

| Error                        | Fix                                                                                                                |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Connection or download error on the first run | VTC could not download the model from Hugging Face. Check your internet connection; on a cluster, see [Offline compute nodes](../advanced/hpc.md#offline-compute-nodes) |
| `ModelNotFoundError`         | A `--checkpoint`, `--config` or `--thresholds-location` path does not contain the expected file (a single `.pt`/`.ckpt` file, a `config.toml`, or `<name>.toml`), or the model version is unknown (valid versions: `2.2`, `2.1`; VTC 2.0 only runs from the `v2.0.2` tag, see [Using VTC 2.0](versions.md#using-vtc-20)). With `--high-precision` and `--thresholds-location`, the folder must contain an `hp.toml` file |
| `CUDA out of memory`         | Reduce batch size, or use `--device cpu`                                                                           |
| `No CUDA GPUs are available` | Use `--device cpu` or install CUDA drivers                                                                         |
| `FileNotFoundError: ... raw_rttm` | No file could be processed: `--wavs` contains no `.wav` files (add `--recursive_search` if they are in subfolders), or every file failed. VTC only reads `.wav` files; convert other formats first (see [Getting Started](../getting-started.md#prepare-your-audio)) |
| `File <name> could not be processed, skipping` | That file was skipped and has no output. Check that it is 16 kHz mono with `ffprobe file.wav`, and convert it if needed (see [Getting Started](../getting-started.md#prepare-your-audio)) |
| Empty output files           | Check that the audio is 16 kHz mono with `ffprobe file.wav`. An empty RTTM file can also mean that no speech was detected in that recording |
| Segments at implausible times, or too few or too many segments | The file is probably not sampled at 16 kHz. VTC does not check the sample rate; convert the file and run VTC again |
| `convert-audios` produces no files | It only converts `.wav` files. Convert other formats with FFmpeg (see [Getting Started](../getting-started.md#prepare-your-audio)) |

## Reporting a problem

Open an issue on [GitHub](https://github.com/LAAC-LSCP/VTC/issues) with:

- your operating system,
- the Python version (`uv run python --version`),
- the exact command you ran, and
- the **full error message**, copied and pasted.
