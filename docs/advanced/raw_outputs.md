---
# icon: lucide/server-cog # gpu
icon: octicons/search-16
---

# Raw model outputs

## What "raw" means

VTC's underlying model produces a frame-by-frame guess about who is speaking. The result is choppy: a single person talking can come out as several short detections with small gaps between them, which isn't very useful for analysis.

So before saving the results, VTC does two cleanups:

1. **Drop very short isolated detections** that are most likely model glitches.
2. **Merge same-speaker detections** that are separated by only a tiny gap (under 100 ms by default) into a single longer segment.

For most users, the cleaned-up output is what you want. But if you're debugging unexpected results, building your own custom post-processing, or studying the model's behaviour itself, the raw output is more informative.

!!! info 
    You can tweak these settings at inference time yourself, head over to the [example commands](../guide.md#example-commands) section in the guide page for more informations.

## How to keep the raw output

Add the `--keep_raw` flag when running VTC. The pre-cleanup detections will be saved alongside the cleaned-up output:

```
<output_folder>/
├── 📂 rttm/          # Cleaned-up segments (default output)
├── 📂 raw_rttm/      # Pre-cleanup, raw model detections (only if --keep_raw)
├── 📄 rttm.csv
└── 📄 raw_rttm.csv
```

The `raw_rttm/` folder follows the same RTTM format as `rttm/`; it just contains more (and shorter) segments.

<!-- For an example on how to use the inference script with different  segment merging parameters, head over to the [example commands](../guide.md#example-commands)  section. -->

---

## Technical Reference

> *Implementation details of the post-processing pipeline.*

The default post-processing pipeline applied to model outputs:

1. **Per-frame binarization** of the four class scores against the active threshold set (`f1.toml` by default; `hp.toml` with `--high_precision`; or any TOML passed via `--thresholds`). See [Threshold selection](thresholds.md).
2. **Drop short segments** of duration less than `--min_duration_on_s` seconds (default `0.1`).
3. **Bridge same-class gaps** of duration less than `--min_duration_off_s` seconds (default `0.1`).

The segment-merging logic is the one from [pyannote.audio](https://github.com/pyannote/pyannote-audio).

Pass `--keep_raw` to additionally write the segments produced after **step 1** but before **steps 2–3** to `<output>/raw_rttm/`. To inspect even earlier, prior to binarization, use `--save_logits` (this can produce large files).
