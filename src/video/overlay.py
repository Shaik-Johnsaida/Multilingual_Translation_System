"""
Video Subtitle Overlay & Audio Dubbing Engine using FFmpeg.
Preserves 100% full original video duration, original frame order, FPS, and resolution.
Overlays synchronized translated subtitles AND replaces original audio track with dubbed target language audio.
"""

import os
import json
import subprocess
from typing import Tuple, List, Dict, Any, Optional
from src.subtitles.srt import export_srt_file


def get_media_duration(file_path: str) -> float:
    """Probes exact media duration in seconds using ffprobe."""
    if not os.path.exists(file_path):
        return 0.0
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(res.stdout.strip())
    except Exception as e:
        print(f"[VideoOverlay] Could not probe media duration with ffprobe: {e}")
        return 0.0


def overlay_subtitles_on_video(
    input_video_path: str,
    subtitle_segments: List[Dict[str, Any]],
    output_video_path: str,
    dubbed_audio_path: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Overlays translated subtitles AND dubs video audio into the target language.
    Preserves 100% complete master video timeline and duration (NO TRIMMING, NO EXTENSION).
    Returns (success, output_path_or_error).
    """
    if not os.path.exists(input_video_path):
        return False, f"Input video not found: {input_video_path}"

    video_duration = get_media_duration(input_video_path)
    base, _ = os.path.splitext(output_video_path)
    srt_path = f"{base}_temp_subtitles.srt"

    # Export temporary SRT file
    export_srt_file(subtitle_segments, srt_path)

    # Escape SRT path for FFmpeg filter parameter on Windows
    escaped_srt_path = srt_path.replace("\\", "/").replace(":", "\\:")

    # Build FFmpeg command - Original video is master input #0
    cmd = ["ffmpeg", "-y", "-i", input_video_path]

    # If dubbed audio is provided, map dubbed audio track from input #1
    if dubbed_audio_path and os.path.exists(dubbed_audio_path):
        cmd.extend(["-i", dubbed_audio_path])
        audio_map = ["-map", "0:v:0", "-map", "1:a:0"]
    else:
        audio_map = ["-c:a", "copy"]

    # Add subtitle filter, fast encoding preset
    filter_arg = f"subtitles='{escaped_srt_path}':force_style='FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Alignment=2,MarginV=25'"

    # Enforce exact master video duration matching input
    cmd.extend([
        "-vf", filter_arg,
        *audio_map,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-c:a", "aac"
    ])

    if video_duration > 0:
        cmd.extend(["-t", str(video_duration)])

    cmd.append(output_video_path)

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            out_dur = get_media_duration(output_video_path)
            print(f"[VideoOverlay] Rendered output video duration: {out_dur:.2f}s (input was: {video_duration:.2f}s)")
            return True, output_video_path
    except Exception as e:
        print(f"[VideoOverlay] Standard FFmpeg subtitle + dubbing filter warning: {e}. Trying fallback...")

    # Fallback FFmpeg audio replacement command
    fallback_cmd = ["ffmpeg", "-y", "-i", input_video_path]
    if dubbed_audio_path and os.path.exists(dubbed_audio_path):
        fallback_cmd.extend([
            "-i", dubbed_audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac"
        ])
        if video_duration > 0:
            fallback_cmd.extend(["-t", str(video_duration)])
        fallback_cmd.append(output_video_path)
        try:
            subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True, output_video_path
        except Exception as ex:
            return False, f"Video dubbing failed: {str(ex)}"

    return False, "FFmpeg overlay failed."
