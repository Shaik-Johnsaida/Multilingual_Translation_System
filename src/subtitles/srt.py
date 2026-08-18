"""
SubRip (.srt) Subtitle Generator & Exporter.
"""

from typing import List, Dict, Any


def generate_srt_content(subtitle_segments: List[Dict[str, Any]], include_speaker: bool = True) -> str:
    """
    Generates standard valid SRT subtitle content string.
    Example:
    1
    00:00:01,200 --> 00:00:04,000
    Speaker 1: Translated text here...
    """
    blocks = []
    for seg in subtitle_segments:
        idx = seg["segment_id"]
        start = seg["start_srt"]
        end = seg["end_srt"]
        text = seg["translated_text"]
        speaker = seg.get("speaker", "")

        if include_speaker and speaker:
            content = f"{speaker}: {text}"
        else:
            content = text

        block = f"{idx}\n{start} --> {end}\n{content}\n"
        blocks.append(block)

    return "\n".join(blocks)


def export_srt_file(subtitle_segments: List[Dict[str, Any]], output_filepath: str) -> str:
    """Writes SRT content to file and returns absolute path."""
    content = generate_srt_content(subtitle_segments)
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return output_filepath
