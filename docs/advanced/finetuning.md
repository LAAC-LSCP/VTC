---
# icon: lucide/server-cog # gpu
icon: octicons/ai-model-24
---

# Fine-tuning VTC 2


Fine-tuning VTC on your own annotated data can improve results for specific recording environments or populations. This requires annotated audio (human-labeled RTTM files), a GPU with 16+ GB memory, and familiarity with deep learning training pipelines.

The fine-tuning code is at [arxaqapi/vtc-finetune](https://github.com/arxaqapi/vtc-finetune). The general workflow is:

1. Prepare annotated data (audio + RTTM) split into train/validation/test
2. Configure training starting from the pre-trained VTC 2.0 checkpoint
3. Train and evaluate against the base model to confirm improvement
