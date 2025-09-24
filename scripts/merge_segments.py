# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "pyannote-audio",
#     "tqdm",
# ]
# ///
from pathlib import Path
import copy

import tqdm
import polars as pl
from pyannote.core import Annotation, Segment


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
        data = pl.DataFrame(
            None, ("uid", "start_time_s", "duration_s", "label")
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", required=True, help="Input folder containing a list of RTTMs.")
    parser.add_argument("-o", "--output", required=True, help="Output folder that will contain the RTTMs with the merged segments.")
    parser.add_argument("--min-duration-on-s", default=0.1, help="Remove speech segments shorter than that many seconds.")
    parser.add_argument("--min-duration-off-s", default=0.1, help="Fill same-speaker gaps shorter than that many seconds.")

    args = parser.parse_args()
    args.folder = Path(args.folder)
    args.output = Path(args.output)

    if not args.folder.exists():
        raise ValueError(f"{args.folder=} does not exists.")
    if not args.output.exists():
        args.output.mkdir(parents=True)

    uri_to_annot: dict[str, Annotation] = {}
    uri_to_proc_annot: dict[str, Annotation] = {}
    for file in tqdm.tqdm(list(args.folder.glob("*.rttm")) + list(args.folder.glob("*.aa"))):
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
            uri_to_proc_annot[uri] = process_annot(annot)

        for uri, annot in uri_to_proc_annot.items():
            (args.output / uri).with_suffix(".rttm").write_text(annot.to_rttm())