from pathlib import Path

from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder


def convert_audios(
    audio_paths: list[Path],
    output: Path,
    allow_upsampling: bool,
    target_sample_rate: int = 16_000,
):
    """Given a list of audio file path, load each file perform downsampling or upsampling to the `target_sample_rate` and convert to mono.

    Args:
        audio_paths (list[Path]): List of Path to the audio files
        output (Path): Output path where the converted audio files will be saved
        allow_upsampling (bool): Mandatory flag if some audios require upsampling
        target_sample_rate (int, optional): Target sample rate. Defaults to 16_000.

    Raises:
        ValueError: Raises if the input audio file is not in `.wav` format
        ValueError: Raises if the sample rate of the audio is lower than the `target_sample_rate`
            and if `allow_upsampling` is not explicitely set to true
    """
    output.mkdir(parents=True, exist_ok=True)

    for audio_p in audio_paths:
        if not audio_p.suffix == ".wav":
            raise ValueError(f"File `{audio_p.name}` is not a wav file.")

        original_sr = AudioDecoder(audio_p).metadata.sample_rate

        if original_sr < target_sample_rate and not allow_upsampling:
            raise ValueError(
                f"File `{audio_p.name}` has a sample rate of {original_sr} Hz, "
                f"which is below the target of {target_sample_rate} Hz. "
                f"Set --allow_upsampling to allow upsampling."
            )

        # Decode with resampling and mono conversion in one step
        decoder = AudioDecoder(
            audio_p,
            sample_rate=target_sample_rate,
            num_channels=1,
        )
        audio_samples = decoder.get_all_samples()

        # Encode and write to disk
        encoder = AudioEncoder(
            samples=audio_samples.data, sample_rate=target_sample_rate
        )
        encoder.to_file(output / audio_p.name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wavs", type=Path, help="input folder containing the audio files to convert."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output folder containing the converted audio files.",
    )
    parser.add_argument("--allow_upsampling", action="store_true")

    args = parser.parse_args()

    convert_audios(
        audio_paths=sorted(list(args.wavs.rglob("*.wav"))),
        output=args.output,
        allow_upsampling=args.allow_upsampling,
    )
