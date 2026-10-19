---
icon: lucide/server-cog
---

# HPC / SLURM setup

This page explains how to run VTC on a compute cluster that uses the SLURM job scheduler. You don't need it to run VTC on your own machine.

## When to use a cluster

A cluster is worth the setup when you have many recordings (tens to hundreds of hours), no fast GPU on your own machine, and access to a cluster through your institution. A SLURM job gives VTC a cluster GPU for as long as it runs.

For smaller datasets, or if your machine has a usable GPU, run VTC directly as described in the [User Guide](../guide.md).

---

## Technical Reference

### Example SLURM submission script

Save the following as `run_vtc.sbatch`, edit the paths, and submit it with `sbatch run_vtc.sbatch`. Adjust the partition, modules and resources to your cluster.

```bash
#!/bin/bash
#SBATCH --job-name=vtc2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 (1)
#SBATCH --mem=32G (2)
#SBATCH --time=04:00:00 (3)
#SBATCH --output=vtc_%j.out

cd /path/to/VTC

uv run vtc-infer \
    --wavs /path/to/audio_files \
    --output /path/to/output \
    --device cuda
```

1. Number of CPU cores to reserve.
2. Amount of memory to reserve.
3. Maximum job duration. Adjust it to the size of your dataset.

### Offline compute nodes

VTC downloads its model from Hugging Face on the first run, but compute nodes often have no internet access. Download the model beforehand from a login node by running VTC once there on a short recording with `--device cpu`. The model is cached in `~/.cache/huggingface/hub/`, and jobs on the compute nodes reuse it.

If your home folder is too small or not shared with the compute nodes, set `HF_HOME` to a shared disk, both when downloading and in the job script. With `HF_HUB_OFFLINE=1` in the job script, Hugging Face uses the cache without trying to connect:

```bash
export HF_HOME=/shared/scratch/$USER/huggingface
export HF_HUB_OFFLINE=1
```

### Sizing the job

Estimate the run time for your GPU from the [speed benchmarks](../guide.md#how-long-will-it-take). The 4-hour `--time` above is generous for small datasets.

### Very large datasets

If a dataset is too large for one job, split the audio files into smaller groups and submit a SLURM array job, with each task processing one group and writing to its own output folder. The tasks run in parallel on several nodes, and a failed task only affects its own group.
