import argparse
from .infer import prepare_vtc_inference, run_vtc_with_params


def vtc_infer_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="2.2",
        help="Local config.toml file/directory, or a known model version key (e.g. '2.1', '2.0').",
    )
    parser.add_argument(
        "--uris", help="Path to a file containing the list of uris to use."
    )
    parser.add_argument(
        "--wavs",
        default="data/debug/wav",
        help="Folder containing the audio files to run inference on.",
    )
    parser.add_argument(
        "--checkpoint",
        default="2.2",
        type=str,
        help="Local checkpoint file/directory, or a known model version key (e.g. '2.1', '2.0').",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output Path to the folder that will contain the final predictions.",
    )
    parser.add_argument(
        "--save_probs",
        action="store_true",
        help="Save averaged per-frame probabilities to disk.",
    )
    parser.add_argument(
        "--stride_pct",
        default=0.25,
        type=float,
        help="Sliding window stride as a fraction of chunk_duration_s (0.5 = 50%% overlap, 0.25 = 75%%).",
    )
    parser.add_argument(
        "--thresholds-location",
        type=str,
        default="2.2",
        help=(
            "Local directory/file containing the thresholds TOML, or a known "
            "model version key. Defaults to wherever --checkpoint resolves to."
        ),
    )
    parser.add_argument(
        "--thresholds-name",
        type=str,
        default="f1",
        help=(
            "Thresholds file to use (without extension), e.g. 'f1' -> "
            "'<location>/thresholds/f1.toml'. Ignored if --high-precision is set."
        ),
    )
    parser.add_argument(
        "--high-precision",
        action="store_true",
        help="Use the high-precision thresholds ('hp'), overriding custom thresholds.",
    )
    parser.add_argument(
        "--min_duration_on_s",
        default=0.1,
        type=float,
        help="Remove speech segments shorter than that many seconds.",
    )
    parser.add_argument(
        "--min_duration_off_s",
        default=0.1,
        type=float,
        help="Fill same-speaker gaps shorter than that many seconds.",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size to use for the forward pass of the model.",
    )
    parser.add_argument(
        "--recursive_search",
        action="store_true",
        help="Recursively search for `.wav` files. Might be slow. Defaults to False.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["gpu", "cuda", "cpu", "mps"],
        help="Size of the batch used for the forward pass in the model.",
    )
    parser.add_argument(
        "--keep_raw",
        action="store_true",
        help="If active, the raw RTTM will be kept and saved to disk in the `<output>/raw_rttm/` folder and a `<output>/raw_rttm.csv` file will be created.",
    )

    args = parser.parse_args()
    params = prepare_vtc_inference(**vars(args))
    run_vtc_with_params(params)
