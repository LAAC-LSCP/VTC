import argparse
import copy
from pathlib import Path

import polars as pl
import torchaudio
from pyannote.core import Annotation, Segment
from segma.inference import run_inference_on_audios


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


def load_rttm(path: Path | str):
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
        data = pl.DataFrame(None, ("uid", "start_time_s", "duration_s", "label"))

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
        file = (raw_output_p / file_uri).with_suffix(".rttm")
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
            (merged_out_p / uri).with_suffix(".rttm").write_text(annot.to_rttm())

    # NOTE - Writting missing rttm files
    if write_empty:
        for uri in set(file_uris_to_merge) - set(uri_to_proc_annot.keys()):
            (merged_out_p / uri).with_suffix(".rttm").touch()


def check_audio_files(uris, wavs) -> None:
    """Fails if the audios are not sampled at 16_000 Hz and contain more than one channel."""
    wavs = Path(wavs)

    if uris:
        with Path(uris).open("r") as uri_f:
            files_to_infer_on = [
                (wavs / uri.strip()).with_suffix(".wav") for uri in uri_f.readlines()
            ]
    else:
        files_to_infer_on = list(wavs.glob("*.wav"))

    for wav_p in sorted(files_to_infer_on):
        info = torchaudio.info(uri=wav_p)
        # NOTE - check that the audio is valid
        if not info.sample_rate == 16_000:
            raise ValueError(
                f"file `{wav_p}` is not samlped at 16 000 hz. Please convert your audio files."
            )
        if not info.num_channels == 1:
            raise ValueError(
                f"file `{wav_p}` has more than one channel. You can average your channels or use another channel reduction technique."
            )


def main(
    output: str,
    uris: list[str] | None = None,
    config: str = "model/config.yml",
    wavs: str = "data/debug/wav",
    checkpoint: str = "model/best.ckpt",
    save_logits: bool = False,
    thresholds: None | dict = None,
    min_duration_on_s: float = 0.1,
    min_duration_off_s: float = 0.1,
    batch_size: int = 128,
    write_empty: bool = True,
    write_csv: bool = True,
):
    """Run sliding inference on the given files and then merges the created segments.

    Args:
        uris (list[str]): list of uris to use for prediction.
        config (str, optional): Config file to be loaded and used for inference. Defaults to "model/config.yml".
        wavs (str, optional): _description_. Defaults to "data/debug/wav".
        checkpoint (str, optional): Path to a pretrained model checkpoint. Defaults to "model/best.ckpt".
        output (str, optional): Output Path to the folder that will contain the final predictions.. Defaults to "".
        save_logits (bool, optional): If the prediction scripts saves the logits to disk, can be memory intensive. Defaults to False.
        thresholds (None | dict, optional): If thresholds dict is given, perform predictions using thresholding.. Defaults to None.
        min_duration_on_s (float, optional): Remove speech segments shorter than that many seconds.. Defaults to .1.
        min_duration_off_s (float, optional): Fill same-speaker gaps shorter than that many seconds.. Defaults to .1.
        batch_size (int): Batch size to use during inference. Defaults to 128.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
    """
    check_audio_files(uris, wavs)

    output = Path(output)

    processed_files = run_inference_on_audios(
        config=config,
        uris=uris,
        wavs=wavs,
        checkpoint=checkpoint,
        output=output,
        thresholds=thresholds,
        batch_size=batch_size,
    )

    merge_segments(
        file_uris_to_merge=[f.stem for f in processed_files],
        output=output,
        min_duration_on_s=min_duration_on_s,
        min_duration_off_s=min_duration_off_s,
        write_empty=write_empty,
    )

    # NOTE - write RTTMs to `csv` files
    if write_csv:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="model/config.yml",
        help="Config file to be loaded and used for inference.",
    )
    parser.add_argument("--uris", help="list of uris to use for prediction")
    parser.add_argument("--wavs", default="data/debug/wav")
    parser.add_argument(
        "--checkpoint",
        default="model/best.ckpt",
        help="Path to a pretrained model checkpoint.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output Path to the folder that will contain the final predictions.",
    )
    parser.add_argument(
        "--save-logits",
        action="store_true",
        help="If the prediction scripts saves the logits to disk, can be memory intensive.",
    )
    parser.add_argument(
        "--thresholds",
        help="If thresholds dict is given, perform predictions using thresholding.",
    )
    parser.add_argument(
        "--min-duration-on-s",
        default=0.1,
        help="Remove speech segments shorter than that many seconds.",
    )
    parser.add_argument(
        "--min-duration-off-s",
        default=0.1,
        help="Fill same-speaker gaps shorter than that many seconds.",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        help="Batch size to use for the forward pass of the model.",
    )

    args = parser.parse_args()

    main(**vars(args))
