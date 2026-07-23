import logging
import shutil
import typing as t
from pathlib import Path

from segma.inference import run_inference_on_audios

from .ressources import (
    VersionStr,
    resolve_model_config_path,
    resolve_model_path,
    resolve_thresholds_path,
)
from .utils import merge_segments, write_rttm_csvs

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    datefmt="%Y.%m.%d %H:%M:%S",
)
logger = logging.getLogger("inference")


class VtcInferenceParams(t.NamedTuple):
    """Resolved, ready-to-run parameters for `run_vtc_inference`."""

    output: Path
    config_path: Path
    checkpoint_path: Path
    thresholds_path: Path | None
    uris: Path | None
    wavs: str
    save_probs: bool
    min_duration_on_s: float
    min_duration_off_s: float
    batch_size: int
    stride_pct: float
    write_empty: bool
    write_csv: bool
    recursive_search: bool
    device: t.Literal["gpu", "cuda", "cpu", "mps"]
    keep_raw: bool


def prepare_vtc_inference(
    output: str,
    uris: Path | None = None,
    config: Path | VersionStr = "2.1",
    wavs: str = "data/debug/wav",
    checkpoint: Path | VersionStr = "2.1",
    save_probs: bool = False,
    high_precision: bool = False,
    thresholds: str | None = "f1",
    thresholds_location: Path | VersionStr = "2.1",
    min_duration_on_s: float = 0.1,
    min_duration_off_s: float = 0.1,
    batch_size: int = 128,
    stride_pct: float = 0.25,
    write_empty: bool = True,
    write_csv: bool = True,
    recursive_search: bool = False,
    device: t.Literal["gpu", "cuda", "cpu", "mps"] = "gpu",
    keep_raw: bool = False,
) -> VtcInferenceParams:
    """Resolve raw arguments into concrete filesystem paths.

    Resolves `config` via `resolve_model_config_path`, `checkpoint` via
    `resolve_model_path`, and the thresholds file (name forced to `"hp"`
    when `high_precision` is set) via `resolve_thresholds_path` against
    `checkpoint`. Creates `output` and copies the resolved thresholds file
    into it.

    Raises:
        ModelNotFoundError: if `config`, `checkpoint`, or `thresholds`
            cannot be resolved.
    """

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = resolve_model_config_path(config)
    checkpoint_path = resolve_model_path(checkpoint)

    # Note: this overrides a custom threshold
    thresholds_path: Path | None = None
    if high_precision:
        thresholds_path = resolve_thresholds_path("hp", "2.1")
    else:
        location = thresholds_location if thresholds_location is not None else checkpoint
        thresholds_name = thresholds if thresholds else "f1"
        thresholds_path = resolve_thresholds_path(thresholds_name, location)


    return VtcInferenceParams(
        output=output_dir,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        thresholds_path=thresholds_path,
        uris=uris,
        wavs=wavs,
        save_probs=save_probs,
        min_duration_on_s=min_duration_on_s,
        min_duration_off_s=min_duration_off_s,
        batch_size=batch_size,
        stride_pct=stride_pct,
        write_empty=write_empty,
        write_csv=write_csv,
        recursive_search=recursive_search,
        device=device,
        keep_raw=keep_raw,
    )


def run_vtc_with_params(params: VtcInferenceParams) -> None:
    """Run VTC inference and write merged (and optionally raw) RTTM/CSV outputs.

    Raises:
        OSError: if raw RTTM cleanup fails.
    """
    logger.info("Running inference on audio files.")
    processed_files = run_inference_on_audios(
        config=params.config_path,
        uris=params.uris,
        wavs=params.wavs,  # pyright: ignore[reportArgumentType]
        checkpoint=params.checkpoint_path,
        output=params.output,
        thresholds=params.thresholds_path,
        batch_size=params.batch_size,
        device=params.device,
        recursive=params.recursive_search,
        save_probs=params.save_probs,
        stride_pct=params.stride_pct,
        logger=logger,
    )

    logger.info("Merging detected speech segments.")
    merge_segments(
        file_uris_to_merge=[f.stem for f in processed_files],
        output=params.output,  # pyright: ignore[reportArgumentType]
        min_duration_on_s=params.min_duration_on_s,
        min_duration_off_s=params.min_duration_off_s,
        write_empty=params.write_empty,
    )

    if not params.keep_raw:
        # NOTE - remove <output>/raw_rttm
        shutil.rmtree(str(Path(params.output / "raw_rttm").absolute()))

    if params.write_csv:
        write_rttm_csvs(params.output, keep_raw=params.keep_raw)

    logger.info(
        f"Inference finished, files can be found here: '{params.output.absolute()}/'"
    )


def run_vtc(*args: t.Any, **kwargs: t.Any) -> None:
    """entry point combining argument resolution and inference."""
    params = prepare_vtc_inference(*args, **kwargs)
    run_vtc_with_params(params)
