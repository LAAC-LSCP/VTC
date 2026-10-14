---
icon: lucide/chart-column-decreasing
---

# Threshold selection

Each speaker category has a threshold that controls how readily VTC assigns it. The default thresholds suit most uses. Change them if you want to favour finding all speech (recall) or avoiding wrong labels (precision).

## What thresholds do

For every frame of audio (a few hundredths of a second), the model outputs four scores between 0 and 1, one per category, that express how likely it is that this category is speaking. VTC marks a category as active in that frame when its score is above the category's **threshold**.

- A **lower threshold** produces more segments: less speech is missed (higher **recall**), but more labels are wrong.
- A **higher threshold** produces fewer segments, but more of them are correct (higher **precision**).

The right choice depends on whether missed speech or false detections matter more for your analysis.

---

## Built-in threshold sets

| Threshold set | Name | When to use it |
|---|---|---|
| **Balanced (default)** | `f1` | General use. Maximises F1, which balances recall and precision. |
| **High precision**     | `hp` | When you would rather miss some speech than get false detections. |

Both sets were tuned on the BabyTrain-2025 validation set. They are downloaded with the model, from the `thresholds/` folder of the [`coml/VTC-2`](https://huggingface.co/coml/VTC-2) Hugging Face repository. To use the high-precision set, add `--high-precision`:

```bash
uv run vtc-infer             \
    --wavs <audio_folder>    \
    --output <output_folder> \
    --high-precision         \
    --device cpu
```

VTC copies the threshold file it used (`f1.toml` or `hp.toml`) into the output folder.

---

## Try it interactively

Drag the sliders, or click on a chart, to set the threshold of each category. The macro-F1 at the top updates as you go.

<iframe src="../assets/visualisations/threshold_plot.html?data=tuning_results.json"
        width="100%" height="700" style="border:0;"></iframe>

!!! note "What the curves show"
    Each curve shows a category's F1 score as its threshold goes from 0 to 1. The default `f1.toml` thresholds are at the peaks of these curves.


---

## Technical Reference

### Custom threshold files

Write your thresholds in a [TOML](https://toml.io/en/) file, such as `my_thresholds/custom.toml`, with one section per class. The values below are those of the default `f1` set:

```toml
[KCHI]
lower_bound = 0.437
upper_bound = 1.0

[OCH]
lower_bound = 0.378
upper_bound = 1.0

[MAL]
lower_bound = 0.429
upper_bound = 1.0

[FEM]
lower_bound = 0.377
upper_bound = 1.0
```

Pass the file's location with `--thresholds-location` and its name, without `.toml`, with `--thresholds-name`:

```bash
uv run vtc-infer             \
    --wavs <audio_folder>    \
    --output <output_folder> \
    --thresholds-location my_thresholds/ \
    --thresholds-name custom \
    --device cpu
```

`--thresholds-location` accepts the `.toml` file itself or a folder. A folder is searched recursively and must contain exactly one file with that name. Without `--thresholds-location`, VTC looks in the same place as the model given by `--checkpoint`. `--high-precision` overrides `--thresholds-name`, so it looks for an `hp.toml` file in `--thresholds-location`. The thresholds file is copied into the output folder under its own name, here `custom.toml`.

### Threshold tuning workflow

To re-tune thresholds on your own annotated data:

1. Run inference on your validation set with `--save_probs` to save the per-frame scores.
2. Run the tuning pipeline, a per-class grid search, against your reference RTTM annotations.
3. The pipeline outputs the per-class thresholds that maximise the chosen objective (macro-F1 by default).

!!! note "Per-class independence"
    Each class's threshold is optimised separately; the search does not account for trade-offs between classes.
