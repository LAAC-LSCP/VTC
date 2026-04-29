---
icon: lucide/bug-off
---

# Troubleshooting

## Installation

| Error                                        | Fix                                                                            |
| -------------------------------------------- | ------------------------------------------------------------------------------ |
| `python3.13: command not found`              | Install via [pyenv](https://github.com/pyenv/pyenv): `pyenv install 3.13`      |
| `uv: command not found`                      | `curl -LsSf https://astral.sh/uv/install.sh \| sh`, then restart your terminal |
| `ffmpeg: command not found`                  | `sudo apt install ffmpeg` (Linux) or `brew install ffmpeg` (macOS)             |
| Model weights missing / empty `VTC-2.0/` dir | `git lfs install && git submodule update --init --recursive`                   |
| `uv sync` fails                              | Ensure you're inside `VTC/` and using Python 3.13+                             |

## Runtime

| Error                        | Fix                                                                                                                |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| CUDA out of memory           | Reduce batch size, or use `--device cpu`                                                                           |
| `No CUDA GPUs are available` | Use `--device cpu` or install CUDA drivers                                                                         |
| No `.wav` files found        | VTC only reads `.wav` — convert other formats first (see [Getting Started](getting-started.md#prepare-your-audio)) |
| Empty output files           | Check audio format (`ffprobe file.wav` — should be 16 kHz, mono) and verify model weights are present              |

## Still stuck?

Open an issue on [GitHub](https://github.com/LAAC-LSCP/VTC/issues) with your OS, Python version, the command you ran, and the full error message.
