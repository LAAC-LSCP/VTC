import typing as t
from pathlib import Path

from huggingface_hub import snapshot_download

VersionStr: t.TypeAlias = str
REPO_ID = "coml/VTC-2"
MODEL_TO_REVISION = {
    "2.2": "91167f6",
    "2.1": "91167f6",
    "2.0": "91e67b5",
}
_WEIGHTS_SUFFIXES = ("*.pt", "*.ckpt")


class ModelResolutionError(Exception):
    """Base error for model-artifact resolution failures."""


class ModelNotFoundError(ModelResolutionError):
    """Raised when a model artifact cannot be resolved from any source."""


def _resolve_named_file(directory_or_file: Path, filename: str) -> Path | None:
    """Return `filename`, if `directory_or_file` is that file or a directory containing it."""
    resolved = directory_or_file
    if resolved.is_file() and resolved.name == filename:
        return directory_or_file
    elif resolved.is_dir():
        matches = list(directory_or_file.rglob(filename))
        if len(matches) != 1:
            return None
        return matches[0]
    return None


def _resolve_pt_path(directory_or_file: Path) -> Path | None:
    """Return the model weights file inside `directory_or_file`, if present.

    Accepts a direct path to a `.pt`/`.ckpt` file, or a directory to search
    for one. If multiple candidates are found, a file whose stem is `best`
    is preferred; otherwise the first match (sorted) is used.
    """
    resolved = directory_or_file
    if resolved.is_file() and resolved.suffix in (".pt", ".ckpt"):
        return directory_or_file
    elif resolved.is_dir():
        matches = sorted(
            match
            for pattern in _WEIGHTS_SUFFIXES
            for match in directory_or_file.rglob(pattern)
        )
        if not matches:
            return None
        for match in matches:
            if match.stem == "best":
                return match
        return matches[0]
    return None


def _download_repo(model_version: VersionStr) -> Path:
    """Fetch (or reuse the cached copy of) the HF Hub snapshot for `model_version`.

    Raises:
        ModelNotFoundError: if `model_version` is not a known MODEL_TO_REVISION key.
    """
    revision = MODEL_TO_REVISION.get(model_version)
    if revision is None:
        raise ModelNotFoundError(f"Unknown model version: {model_version!r}")
    return Path(snapshot_download(repo_id="coml/VTC-2", revision=revision))


def resolve_model_path(model: Path | VersionStr) -> Path:
    """Resolve a model reference to its `.pt | .ckpt` weights file.

    Tries, in order: `model` as a local directory containing a single `.pt | .ckpt`
    file; `model` as a string naming such a directory; `model` as a
    MODEL_TO_REVISION key, in which case the HF Hub repo is fetched and
    `model/best.ckpt` is used.

    Raises:
        ModelNotFoundError: if none of the above resolve to an existing file.
    """
    if isinstance(model, Path):
        pt_file = _resolve_pt_path(model)
        if pt_file is None:
            raise ModelNotFoundError(f"No single .pt or .ckpt file found in {model}")
        return pt_file

    local_dir = Path(model)
    if local_dir.is_dir() or local_dir.is_file():
        pt_file = _resolve_pt_path(local_dir)
        if pt_file is not None:
            return pt_file

    repo_dir = _download_repo(model)
    pt_file = repo_dir / "model" / "best.ckpt"
    if not pt_file.is_file():
        raise ModelNotFoundError(f"{pt_file} not found in downloaded repo")
    return pt_file


def resolve_model_config_path(model: Path | VersionStr) -> Path:
    """Resolve a model reference to its `config.toml`.

    Tries, in order: `model` as a direct path to `config.toml` or a local
    directory containing it; `model` as a string naming the same; `model`
    as a MODEL_TO_REVISION key, in which case the HF Hub repo is fetched
    and `model/config.toml` is used.

    Raises:
        ModelNotFoundError: if none of the above resolve to an existing file.
    """
    if isinstance(model, Path):
        config_file = _resolve_named_file(model, "config.toml")
        if config_file is None:
            raise ModelNotFoundError(f"No config.toml found in {model}")
        return config_file

    local_path = Path(model)
    if local_path.is_dir() or local_path.is_file():
        config_file = _resolve_named_file(local_path, "config.toml")
        if config_file is not None:
            return config_file

    repo_dir = _download_repo(model)
    config_file = repo_dir / "model" / "config.toml"
    if not config_file.is_file():
        raise ModelNotFoundError(f"{config_file} not found in downloaded repo")
    return config_file


def resolve_thresholds_path(name: str, location: Path | VersionStr) -> Path:
    """Resolve a thresholds file named `name` relative to `location`.

    Tries, in order: `location` as a direct path to `{name}.toml` or a local
    directory containing it; `location` as a string naming the same;
    `location` as a MODEL_TO_REVISION key, in which case the HF Hub repo is
    fetched and `thresholds/{name}.toml` is used.

    Raises:
        ModelNotFoundError: if none of the above resolve to an existing file.
    """
    filename = f"{name}.toml"
    if isinstance(location, Path):
        thresholds_file = _resolve_named_file(location, filename)
        if thresholds_file is None:
            raise ModelNotFoundError(f"No {filename} found in {location}")
        return thresholds_file

    local_path = Path(location)
    if local_path.is_dir() or local_path.is_file():
        thresholds_file = _resolve_named_file(local_path, filename)
        if thresholds_file is not None:
            return thresholds_file

    repo_dir = _download_repo(location)
    thresholds_file = repo_dir / "thresholds" / filename
    if not thresholds_file.is_file():
        raise ModelNotFoundError(f"{thresholds_file} not found in downloaded repo")
    return thresholds_file
