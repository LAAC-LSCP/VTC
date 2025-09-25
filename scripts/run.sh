# NOTE - set 'audios_path' as the folder containing the audio files
# NOTE - set 'output' as the output folder that will contain the RTTM files

audios_path=audio_folder
output=out/$(date +%Y%m%d_%H%M)_inference_output

mkdir -p $output

uv run scripts/infer.py \
    --wavs $audios_path \
    --output $output
