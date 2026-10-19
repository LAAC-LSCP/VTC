import copy
from pathlib import Path

import polars as pl
from pyannote.core import Annotation, Segment
from segma.utils.io import get_audio_info


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


def write_rttm_csvs(output: Path, *, keep_raw: bool) -> None:
    """Load merged (and optionally raw) RTTM files and write them out as CSVs."""
    if keep_raw:
        raw_rttm_files = sorted((output / "raw_rttm").glob("*.rttm"))
        raw_rttm_dfs = [load_rttm(f) for f in raw_rttm_files]
        pl.concat(raw_rttm_dfs).write_csv(output / "raw_rttm.csv")

    rttm_files = sorted((output / "rttm").glob("*.rttm"))
    rttm_dfs = [load_rttm(f) for f in rttm_files]
    pl.concat(rttm_dfs).write_csv(output / "rttm.csv")
