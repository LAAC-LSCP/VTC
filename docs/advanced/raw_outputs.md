---
# icon: lucide/server-cog # gpu
icon: octicons/search-16
---

# Raw model outputs

## What "raw" means

The model makes a decision for every short frame of audio, so its output is fragmented: one person speaking can appear as several short segments separated by small gaps. Before writing the results, VTC cleans them up in two steps:

1. **Merge segments of the same speaker type** separated by a gap shorter than 0.1 s.
2. **Drop segments shorter than 0.1 s**, which are most likely errors.

The cleaned output suits most analyses. The raw output is useful for debugging unexpected results, writing your own post-processing, or studying the model itself.

!!! info
    Both durations can be changed when running VTC; see the [example commands](../guide.md#example-commands).

## Keeping the raw output

Add the `--keep_raw` flag. The raw segments are then saved next to the cleaned ones:

```
<output_folder>/
├── 📂 rttm/          # Cleaned-up segments (default output)
├── 📄 rttm.csv
├── 📂 raw_rttm/      # Raw model segments (only with --keep_raw)
└── 📄 raw_rttm.csv   # (only with --keep_raw)
```

`raw_rttm/` uses the same RTTM format as `rttm/`, with more and shorter segments.

<!-- For an example on how to use the inference script with different  segment merging parameters, head over to the [example commands](../guide.md#example-commands)  section. -->

---

## Technical Reference

Processing steps applied to the model output:

1. **Sliding-window averaging.** The model runs on overlapping windows (step set by `--stride_pct`, default `0.25` of the window length), and the per-frame scores of overlapping windows are averaged.
2. **Per-frame binarization** of the four class scores with the active threshold set: `f1.toml` by default, `hp.toml` with `--high-precision`, or a custom file selected with `--thresholds-location` and `--thresholds-name`. See [Threshold selection](thresholds.md).
3. **Gap filling:** same-class segments separated by less than `--min_duration_off_s` seconds (default `0.1`) are merged.
4. **Short-segment removal:** segments shorter than `--min_duration_on_s` seconds (default `0.1`) are dropped.

Steps 3 and 4 use [pyannote.audio](https://github.com/pyannote/pyannote-audio) annotation operations.

`--keep_raw` writes the segments obtained after step 2 to `<output>/raw_rttm/`. `--save_probs` writes the averaged per-frame probabilities from step 1 to `<output>/probs/<file>-probs_dict_t.pt`, a PyTorch file with one tensor per class. These files can be large.
