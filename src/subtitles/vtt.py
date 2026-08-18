"""
WebVTT (.vtt) Subtitle Generator & Exporter.
"""

from typing import List, Dict, Any


def generate_vtt_content(subtitle_segments: List[Dict[str, Any]], include_speaker: bool = True) -> str:
    """
    Generates standard valid WebVTT (.vtt) subtitle content string.
    Example:
    WEBVTT

    00:00:01.200 --> 00:00:04.000
    <v Speaker 1>Translated text...
    """
    lines = ["WEBVTT\n"]
    for seg in subtitle_segments:
        start = seg["start_vtt"]
        end = seg["end_vtt"]
        text = seg["translated_text"]
        speaker = seg.get("speaker", "")

        if include_speaker and speaker:
            content = f"<v {speaker}>{text}"
        else:
            content = text

        lines.append(f"{start} --> {end}\n{content}\n")

    return "\n".join(lines)


def export_vtt_file(subtitle_segments: List[Dict[str, Any]], output_filepath: str) -> str:
    """Writes WebVTT content to file and returns absolute path."""
    content = generate_vtt_content(subtitle_segments)
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return output_filepath
