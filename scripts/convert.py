from pathlib import Path

import torchaudio


def convert_audios(audio_paths: list[Path], output: Path, allow_upsampling: bool):
    for audio_p in audio_paths:
        if not audio_p.suffix == ".wav":
            raise ValueError(f"File `{audio_p.name}` is not a wav file.")

        info = torchaudio.info(uri=audio_p)
        audio_t = torchaudio.load(uri=audio_p.resolve())[0]

        if info.sample_rate > 16_000:
            audio_t = torchaudio.functional.resample(
                audio_t, orig_freq=info.sample_rate, new_freq=16_000
            )
        elif info.sample_rate < 16_000 and allow_upsampling:
            audio_t = torchaudio.functional.resample(
                audio_t, orig_freq=info.sample_rate, new_freq=16_000
            )
        if info.num_channels > 1:
            audio_t = audio_t.mean(0, keepdim=True)

        # NOTE - write audio to disk
        torchaudio.save(output / audio_p.name, src=audio_t)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wavs", help="input folder containing the audio files to convert."
    )
    parser.add_argument(
        "--output", help="Output folder containing the converted audio files."
    )
    parser.add_argument("--allow_upsampling", action="store_true")

    args = parser.parse_args()

    convert_audios(
        audio_paths=sorted(list(Path(args.wavs).glob("*.wav"))),
        output=args.output,
        allow_upsampling=args.allow_upsampling,
    )
