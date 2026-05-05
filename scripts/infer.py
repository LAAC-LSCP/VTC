import argparse
import copy
import logging
import shutil
from pathlib import Path
from typing import Literal

import polars as pl
from huggingface_hub import snapshot_download
from pyannote.core import Annotation, Segment
from segma.inference import get_list_of_files_to_process, run_inference_on_audios
from segma.utils.io import get_audio_info

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    datefmt="%Y.%m.%d %H:%M:%S",
)
logger = logging.getLogger("inference")

MODEL_TO_REVISION = {
    "2.1": "10265c5",
    "2.0": "91e67b5",
}


def load_aa(path: Path):
    data = pl.read_csv(
        source=path,
        has_header=False,
        new_columns=("uid", "start_time_s", "duration_s", "label"),
        schema={
            "uid": pl.String(),
            "start_time_s": pl.Float64(),
            "duration_s": pl.Float64(),
            "label": pl.String(),
        },
        separator=" ",
    )
    return data


def load_rttm(path: Path | str) -> pl.DataFrame:
    try:
        data = pl.read_csv(
            source=path,
            has_header=False,
            columns=[1, 3, 4, 7],
            new_columns=("uid", "start_time_s", "duration_s", "label"),
            schema_overrides={
                "uid": pl.String(),
                "start_time_s": pl.Float64(),
                "duration_s": pl.Float64(),
                "label": pl.String(),
            },
            separator=" ",
        )
    except pl.exceptions.NoDataError:
        data = pl.DataFrame(
            None,
            {
                "uid": pl.String(),
                "start_time_s": pl.Float64(),
                "duration_s": pl.Float64(),
                "label": pl.String(),
            },
        )

    return data


def load_one_uri(uri_df: pl.DataFrame):
    for uids, turns in uri_df.group_by("uid"):
        uid = uids[0]
        annotation = Annotation(uri=uid)
        for i, turn in enumerate(turns.iter_rows(named=True)):
            segment = Segment(
                turn["start_time_s"], turn["start_time_s"] + turn["duration_s"]
            )
            annotation[segment, i] = turn["label"]
        yield uid, annotation


def process_annot(
    annotation: Annotation,
    min_duration_off_s: float = 0.1,
    min_duration_on_s: float = 0.1,
) -> Annotation:
    """Create a new `Annotation` with the `min_duration_off` and `min_duration_off` rules applied.

    Args:
        annotation (Annotation): input annotation
        min_duration_off_s (float, optional): Remove speech segments shorter than that many seconds. Defaults to 0.1.
        min_duration_on_s (float, optional): Fill same-speaker gaps shorter than that many seconds. Defaults to 0.1.

    Returns:
        Annotation: Processed annotation.
    """
    active = copy.deepcopy(annotation)
    # NOTE - Fill regions shorter than that many seconds.
    if min_duration_off_s > 0.0:
        active = active.support(collar=min_duration_off_s)
    # NOTE - remove regions shorter than that many seconds.
    if min_duration_on_s > 0:
        for segment, track in list(active.itertracks()):
            if segment.duration < min_duration_on_s:
                del active[segment, track]
    return active


def merge_segments(
    file_uris_to_merge: list[str],
    output: str,
    min_duration_on_s: float = 0.1,
    min_duration_off_s: float = 0.1,
    write_empty: bool = True,
):
    output = Path(output)
    raw_output_p = output / "raw_rttm"

    # NOTE - merge RTTMs
    uri_to_annot: dict[str, Annotation] = {}
    uri_to_proc_annot: dict[str, Annotation] = {}
    merged_out_p = output / "rttm"
    merged_out_p.mkdir(exist_ok=True, parents=True)

    for file_uri in file_uris_to_merge:
        file = raw_output_p / f"{file_uri}.rttm"
        if not file.exists():
            continue

        match file.suffix:
            case ".aa":
                data = load_aa(file)
            case ".rttm":
                data = load_rttm(file)
            case _:
                raise ValueError(
                    f"File not found error, extension is not supported: {file}"
                )

        # NOTE - process, should handle the case where a single rttm contains multiple URIS
        for uri, annot in load_one_uri(data):
            uri_to_annot[uri] = annot
            uri_to_proc_annot[uri] = process_annot(
                annotation=annot,
                min_duration_off_s=min_duration_off_s,
                min_duration_on_s=min_duration_on_s,
            )

        for uri, annot in uri_to_proc_annot.items():
            (merged_out_p / f"{uri}.rttm").write_text(annot.to_rttm())

    # NOTE - Writting missing rttm files
    if write_empty:
        for uri in set(file_uris_to_merge) - set(uri_to_proc_annot.keys()):
            (merged_out_p / f"{uri}.rttm").touch()


def check_audio_files(audio_files_to_process: list[Path]) -> None:
    """Fails if the audios are not sampled at 16_000 Hz and contain more than one channel."""

    for wav_p in audio_files_to_process:
        info = get_audio_info(wav_p)
        # NOTE - check that the audio is valid
        if not info.sample_rate == 16_000:
            raise ValueError(
                f"file `{wav_p}` is not samlped at 16 000 hz. Please convert your audio files."
            )
        if not info.n_channels == 1:
            raise ValueError(
                f"file `{wav_p}` has more than one channel. You can average your channels or use another channel reduction technique."
            )


def main(
    wavs: str,
    output: str | Path,
    model: Literal["2.1"] = "2.1",
    device: Literal["gpu", "cuda", "cpu", "mps"] = "gpu",
    batch_size: int = 128,
    recursive_search: bool = False,
    high_precision: bool = False,
    thresholds: None | Path = Path("thresholds/f1.toml"),
    min_duration_on_s: float = 0.1,
    min_duration_off_s: float = 0.1,
    write_empty: bool = True,
    save_logits: bool = False,
    keep_raw: bool = False,
    uris: Path | None = None,
    *args,
    **kwargs,
):
    """Run sliding inference on the given files and then merges the created segments.

    Args:
        wavs (str): List of audio files to run inference on.
        output (str | Path): Output Path to the folder that will contain the final predictions.
        model (Literal[&quot;2.1&quot;], optional): Version of the VTC model to use. Defaults to "2.1".
        device (Literal[&quot;gpu&quot;, &quot;cuda&quot;, &quot;cpu&quot;, &quot;mps&quot;], optional): Device to run the model on. Defaults to "gpu".
        batch_size (int, optional): Batch size to use during inference. Defaults to 128.
        recursive_search (bool, optional): Recursively searches the wavs dir for audio files, can be time consuming. Defaults to False.
        high_precision (bool, optional): Uses a high precision version of the VTC by using the high precision thresholds (`thresholds/hp.toml`). Defaults to False.
        thresholds (None | Path, optional): Path to a thresholds dict, perform predictions using thresholding. Defaults to Path("thresholds/f1.toml").
        min_duration_on_s (float, optional): Remove speech segments shorter than that many seconds. Defaults to .1.
        min_duration_off_s (float, optional): Fill same-speaker gaps shorter than that many seconds. Defaults to .1.
        write_empty (bool, optional): Write an empty RTTM files to disk when nothing was detected in an audio. Defaults to True.
        save_logits (bool, optional): If the prediction scripts saves the logits to disk, can be memory intensive. Defaults to False.
        keep_raw (bool, optional): If True, keeps the RTTM files before segment merging. Defaults to False.
        uris (Path | None, optional): Given a `.txt` file containing a list of newline separated uris (filenames), performs inference only on these audio files. Defaults to None.

    Raises:
        FileNotFoundError: Raised if the model checkpoint and config file where not correctly downloaded using `huggingface_hub.snapshot_download`.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using VTC model version {model}.")
    model_path = snapshot_download(
        repo_id="coml/VTC-2",
        revision=MODEL_TO_REVISION[model],
    )

    config_path = Path(model_path) / "model" / "config.toml"
    checkpoint_path = Path(model_path) / "model" / "best.ckpt"
    if not config_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Something went wrong when downloading the model, the files could not be found on disk:\n\tPath: {model_path}"
        )

    logger.info(f"VTC-{model} model is stored here: {model_path}.")

    # TODO - use assets
    if high_precision:
        thresholds = Path("thresholds/hp.toml")
    if thresholds:
        shutil.copy(str(thresholds), dst=output)
        logger.info(f"Using thresholds: {thresholds}")

    logger.info("Running inference on audio files...")
    processed_files = run_inference_on_audios(
        config=config_path,
        uris=uris,
        wavs=wavs,
        checkpoint=checkpoint_path,
        output=output,
        thresholds=thresholds,
        batch_size=batch_size,
        device=device,
        recursive=recursive_search,
        save_logits=save_logits,
        logger=logger,
    )

    logger.info("Merging detected speech segments.")
    merge_segments(
        file_uris_to_merge=[f.stem for f in processed_files],
        output=output,
        min_duration_on_s=min_duration_on_s,
        min_duration_off_s=min_duration_off_s,
        write_empty=write_empty,
    )

    if not keep_raw:
        # NOTE - remove <output>/raw_rttm
        shutil.rmtree(str(Path(output / "raw_rttm").absolute()))

    # NOTE - write RTTMs to `csv` files
    if keep_raw:
        # NOTE - Raw RTTMs
        raw_rttm_file_p = sorted(list((output / "raw_rttm").glob("*.rttm")))
        raw_rttm_file_dfs = []
        for rttm_file in raw_rttm_file_p:
            raw_rttm_file_dfs.append(load_rttm(rttm_file))
        pl.concat(raw_rttm_file_dfs).write_csv(output / "raw_rttm.csv")

    # NOTE - merged RTTMs
    rttm_file_p = sorted(list((output / "rttm").glob("*.rttm")))
    rttm_file_dfs = []
    for rttm_file in rttm_file_p:
        rttm_file_dfs.append(load_rttm(rttm_file))
    pl.concat(rttm_file_dfs).write_csv(output / "rttm.csv")

    logger.info(f"Inference finished, files can be found here: '{output.absolute()}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="2.1",
        choices=["2.1"],  # "2.0"
        help="Version of the model to use during inference. Defaults to `2.1`",
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
        "--output",
        required=True,
        type=Path,
        help="Output Path to the folder that will contain the final predictions.",
    )
    parser.add_argument(
        "--save_logits",
        action="store_true",
        help="If the prediction scripts saves the logits to disk, can be memory intensive.",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("thresholds/f1.toml"),
        help="If thresholds dict is given, perform predictions using thresholding.",
    )
    parser.add_argument(
        "--high_precision",
        action="store_true",
        help="Loads the high precision thresholds, overwrites the `--thresholds` argument.",
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

    main(**vars(args))
