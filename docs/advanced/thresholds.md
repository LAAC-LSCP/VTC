---
icon: lucide/chart-column-decreasing
---

# Threshold selection

> **VTC has a tunable "sensitivity" setting for each speaker category. The default settings work well for most users; this page is for those who want to deliberately trade off catching every segment ("recall") against avoiding wrong labels ("precision").**

## What thresholds actually do

For every short slice of audio (a few hundredths of a second), VTC produces four numbers between 0 and 1, one per speaker category, representing how confident the model is that the category is currently speaking. To turn those continuous scores into yes/no decisions ("Was the key child speaking at this moment? Yes or no?"), VTC compares each score against a **threshold**. If the score is above the threshold, the category is marked as active for that slice; otherwise it is not.

You can think of the threshold as the bar a category has to clear before VTC will commit to a label:

- **Lower threshold** → easier to clear → VTC labels *more* segments. Good if you don't want to miss anything (high **recall**), but you'll get more wrong labels too.
- **Higher threshold** → harder to clear → VTC labels *fewer* segments, but the ones it does label are more likely to be correct (high **precision**).

There is no universally best threshold, the right value depends on whether you care more about avoiding misses (favouring lower thresholds) or avoiding false alarms (favouring higher thresholds).

---

## Built-in threshold sets

VTC ships with two pre-tuned threshold sets. The default balances recall and precision; the high-precision set favours fewer wrong labels at the cost of missing more speech.

| Threshold set | File | When to use it |
|---|---|---|
| **Balanced (default)** | `thresholds/f1.toml` | General use — maximises overall F1 (a balance of recall and precision). |
| **High precision**     | `thresholds/hp.toml` | When you would rather miss some speech than risk introducing false detections. |

Both threshold sets were tuned on the BabyTrain-2025 validation set. The balanced set is selected automatically; to use the high-precision set, add the `--high_precision` flag, or pass the file directly with `--thresholds`:

```bash
uv run scripts/infer.py      \
    --wavs <audio_folder>    \
    --output <output_folder> \
    --high_precision \
    --device cpu
```

---

## Try it interactively

> **The widget below lets you move each threshold up and down and watch the effect on overall accuracy in real time. This is the easiest way to build intuition for what thresholds do.**

Drag each slider, or click directly on a chart, to set the decision threshold for each category. The macro-F1 number at the top updates live as you tune.

<iframe src="../assets/visualisations/threshold_plot.html?data=tuning_results.json"
        width="100%" height="700" style="border:0;"></iframe>

!!! note "What the curves show"
    Each curve traces how the F1 score for one category changes as you sweep its threshold from 0 to 1. The peak of each curve is the value that maximises that category's F1 in isolation; the default `f1.toml` thresholds correspond to those peaks.


---

## Technical Reference

> *Implementation details for engineers tuning their own thresholds.*

### Custom threshold files

To use your own thresholds, create a file `custom.toml` following the [TOML specification](https://toml.io/en/) and pass it to the inference script with `--thresholds`:

```bash
uv run scripts/infer.py      \
    --wavs <audio_folder>    \
    --output <output_folder> \
    --thresholds thresholds/custom.toml \
    --device cpu
```

### Threshold tuning workflow

To re-tune thresholds on your own annotated data:

1. Run inference on your validation set with `--save_logits` to dump per-frame model scores.
2. Run the tuning pipeline (a per-class grid search) against your reference RTTM annotations.
3. The pipeline emits the per-class thresholds maximising your chosen objective (default: macro-F1).

!!! note "Per-class independence"
    Each class's threshold is optimised independently — the search does not jointly account for cross-class trade-offs.
