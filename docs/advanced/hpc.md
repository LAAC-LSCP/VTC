---
icon: lucide/server-cog
---

# HPC / SLURM setup

For more technical users that have access to a compute cluster with slurm, we recommend using slurm to create jobs to run VTC:

```bash
#!/bin/bash
#SBATCH --job-name=vtc2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=vtc_%j.out

cd /path/to/VTC

uv run scripts/infer.py \
    --wavs /path/to/audio_files \
    --output /path/to/output \
    --device cuda
```

Adjust partition names, modules, and resource requests for your cluster. Use the [speed benchmarks](???) to estimate wall time. For very large corpora, we suggest you split the audios into batches and use SLURM array jobs.


