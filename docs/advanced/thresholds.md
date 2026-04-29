---
icon: lucide/chart-column-decreasing
---

<!-- !!! warning "Under construction" -->


# Threshold selection

To make a decision the model outputs values between 0 and 1, one per class. The values are then binarized (i.e. classes which values that reach a certain threshold are set as active).

The model by default uses the thresholds in the `f1.toml` files optimized on the BabyTrain-2025 validation set.

For a high-precision version, you can use the `hp.toml` thresholds using the `--high_precision` flag or by passing the threholds file `--thresholds thresholds/hp.toml` when running the model. These thresholds have been selected by maximizing the precision while ensuring at least 50% of recall for the given class.

```bash
uv run scripts/infer.py      \
    --wavs <audio_folder>    \
    --output <output_folder> \
    --high_precision \
    --device cpu
```

You can use the following interactive visualizer by dragging each slider, or clicking directly on a chart, to set the decision threshold per class.
The macro-F1 at the top updates live as you tune.

<iframe src="../../assets/visualisations/threshold_plot.html?data=tuning_results.json"
        width="100%" height="700" style="border:0;"></iframe>

<!-- For manual threholds selection, you can refer to the following figure, create your own threshold file and use it 

<!-- Insert ~nice~ figure of threholds -->

<!-- 
## Tuning the model on your own data
A simple way to adapt the model to your own data is to tune the thresholds on your own data.
For that you will need a small set of annotated data and then use the inference script to generate logits and then run the tuning pipeline that will grid search over candidate threhsolds and outpout the ones maximizing the average F1 score over all classes.

!!! note 
    Each threshold is independent and does not depend on the other classes.


## Using custom thresholds
Simply create a threshold file `custom.toml` that follows the [toml specifications you can find here](https://toml.io/en/) and pass the file as an argument to the inference script.

```bash
uv run scripts/infer.py      \
    --wavs <audio_folder>    \
    --output <output_folder> \
    --thresholds thresholds/custom.toml \
    --device cpu
```

You can use the following interactive visualizer by dragging each slider, or clicking directly on a chart, to set the decision threshold per class.
The macro-F1 at the top updates live as you tune.

<iframe src="/assets/visualisations/threshold_plot.html?data=tuning_results.json"
        width="100%" height="660" style="border:0;"></iframe> -->