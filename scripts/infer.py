import argparse
import copy
from pathlib import Path

import polars as pl
import torch
import tqdm
import yaml
from pyannote.core import Annotation, Segment
from segma.config import Config, load_config
from segma.models import Models
from segma.predict import sliding_prediction
from segma.utils.encoders import MultiLabelEncoder


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
):
    """Run sliding prediction and merges the created segments.

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

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
    """
    wavs = Path(wavs)
    checkpoint = Path(checkpoint)
    output = Path(output)
    raw_output_p = output / "raw_rttm"

    if thresholds is not None and Path(thresholds).exists():
        with Path(thresholds).open("r") as f:
            threshold_dict = yaml.safe_load(f)
        print(f"[log] - Treshold loaded: {threshold_dict}")
    else:
        threshold_dict = None

    if not wavs.exists():
        raise ValueError(f"Path `{wavs=}` does not exists")
    if not checkpoint.exists():
        raise ValueError(f"Path `{checkpoint=}` does not exists")

    cfg: Config = load_config(config)

    l_encoder = MultiLabelEncoder(labels=cfg.data.classes)

    model = Models[cfg.model.name].load_from_checkpoint(
        checkpoint_path=checkpoint, label_encoder=l_encoder, config=cfg, train=False
    )

    # NOTE - get device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    model = torch.compile(model)

    # NOTE if uris: path is known
    if uris:
        with Path(uris).open("r") as uri_f:
            uris = [uri.strip() for uri in uri_f.readlines()]
        n_files = len(uris)
        for i, uri in enumerate(uris):
            wav_f = (wavs / uri).with_suffix(".wav")
            print(
                f"[log] - ({i:>{len(str(n_files))}}/{n_files}) - running inference for file: '{wav_f.stem}'"
            )
            sliding_prediction(
                wav_f,
                model=model,
                output_p=output,
                config=cfg,
                save_logits=save_logits,
                thresholds=threshold_dict,
            )
    else:
        if wavs.suffix == ".wav" and wavs.is_file():
            wav_files = [wavs]
        else:
            wav_files = list(wavs.glob("*.wav"))
        n_files = len(wav_files)
        for i, wav_f in enumerate(wav_files):
            print(
                f"[log] - ({i:>{len(str(n_files))}}/{n_files}) - running inference for file: '{wav_f.stem}'"
            )
            sliding_prediction(
                wav_f,
                model=model,
                output_p=output,
                config=cfg,
                save_logits=save_logits,
                thresholds=threshold_dict,
            )

    # TODO - warning when file not found but do not fail please
    # NOTE - merge RTTMs
    uri_to_annot: dict[str, Annotation] = {}
    uri_to_proc_annot: dict[str, Annotation] = {}
    merged_out_p = output / "rttm"
    merged_out_p.mkdir(exist_ok=True, parents=True)
    for file in tqdm.tqdm(
        list(raw_output_p.glob("*.rttm")) + list(raw_output_p.glob("*.aa"))
    ):
        if not file.exists():
            import warnings

            warnings.warn(f"File '{file}' does not exist, skipping.")
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

    args = parser.parse_args()

    main(**vars(args))
