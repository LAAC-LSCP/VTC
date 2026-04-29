---
# icon: lucide/server-cog # gpu
icon: octicons/search-16
---

# Understanding raw RTTM model outputs

By default, the inference pipeline applies a post-processing step that removes isolated short segments and merges same-class detections separated by less than 100ms. The raw (pre-post-processing) RTTM files are discarded after the pipeline runs to keep the output tidy.

To retain them, pass the `--keep_raw` flag during inference. The raw segments will be written to `📂 raw_rttm/`, which is useful for debugging or building custom pipelines.

