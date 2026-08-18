"""
Audio Extraction Pipeline using FFmpeg.
Extracts 16kHz mono WAV audio from any input video or audio file.
"""

import os
import subprocess
from typing import Tuple, Optional


def extract_audio(input_file: str, output_wav_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Extracts audio from video/audio file and converts to 16kHz mono PCM WAV.
    Returns (success, output_wav_path_or_error).
    """
    if not os.path.exists(input_file):
        return False, f"Input file not found: {input_file}"

    if not output_wav_path:
        base, _ = os.path.splitext(input_file)
        output_wav_path = f"{base}_extracted_16k.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav_path
    ]

    try:
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 0:
            return True, output_wav_path
        else:
            return False, "FFmpeg created empty audio output."
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode("utf-8", errors="ignore")
        return False, f"FFmpeg extraction failed: {err_msg[:200]}"
    except Exception as e:
        return False, f"Extraction exception: {str(e)}"
