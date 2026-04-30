---
icon: lucide/server-cog
---

# HPC / SLURM setup

> **If your institution has a shared compute cluster (the kind where you submit "jobs" rather than running things directly), this page shows how to run VTC there. If you don't know what SLURM is, you can skip this page — it's not needed for ordinary use.**

## When is this useful?

A shared compute cluster is worth the extra setup when:

- you have **a lot of recordings** (tens to hundreds of hours), and
- you don't have a fast GPU on your own machine, but
- your university or research institute has a cluster you can access.

In that situation, submitting a SLURM job lets you borrow a GPU on the cluster for as long as VTC needs to run, without tying up your laptop.

If your dataset is small or your laptop already has a usable GPU, just run VTC directly as described in the [User Guide](../guide.md).

---

## Technical Reference

### Example SLURM submission script

Save the following as `run_vtc.sbatch`, edit the paths, then submit it with `sbatch run_vtc.sbatch`. Adjust partition names, modules, and resource requests to match your cluster's conventions.

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

uv run scripts/infer.py \
    --wavs /path/to/audio_files \
    --output /path/to/output \
    --device cuda
```

1. Number of CPU cores to reserve on the machine
2. Amount of memory to use
3. Maximum job duration, adjust to match your dataset size.



### Sizing the job

Use the [speed benchmarks](../guide.md#how-long-will-it-take) in the User Guide to estimate wall time for the GPU you're targeting. The 4-hour `--time` value above is conservative; tighten it for shorter corpora.

### Very large corpora

For corpora that won't comfortably finish in a single job, split the audio files across many smaller batches and submit them as a SLURM array job — each array task processes its own slice of the input folder and writes to its own output subfolder. This parallelises across cluster nodes and is also more resilient to individual job failures.
