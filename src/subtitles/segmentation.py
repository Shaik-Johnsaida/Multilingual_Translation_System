"""
Subtitle Segmentation Engine.
Formats transcript & translation segments into optimized, reading-speed adjusted subtitle units.
Locks display timestamps to exact speech audio timing without forward time drift.
"""

from typing import List, Dict, Any, Optional


def format_seconds_to_srt_timestamp(seconds: float) -> str:
    """Formats float seconds into SRT timestamp format: HH:MM:SS,mmm"""
    millis = int((seconds - int(seconds)) * 1000)
    seconds_int = int(seconds)
    hours = seconds_int // 3600
    minutes = (seconds_int % 3600) // 60
    secs = seconds_int % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_seconds_to_vtt_timestamp(seconds: float) -> str:
    """Formats float seconds into WebVTT timestamp format: HH:MM:SS.mmm"""
    millis = int((seconds - int(seconds)) * 1000)
    seconds_int = int(seconds)
    hours = seconds_int // 3600
    minutes = (seconds_int % 3600) // 60
    secs = seconds_int % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def optimize_subtitle_segments(
    raw_segments: List[Dict[str, Any]],
    max_line_chars: int = 42,
    max_duration: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Adjusts subtitle display boundaries and wraps long lines.
    Locks start timestamps to exact speech timing and prevents forward drift.
    """
    if not raw_segments:
        return []

    optimized = []

    for idx, seg in enumerate(raw_segments):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 2.0))

        if max_duration and start >= max_duration:
            continue

        if max_duration and end > max_duration:
            end = max_duration

        if end <= start:
            end = start + 0.8

        # Prevent overlap with previous segment without drifting start time
        if optimized and optimized[-1]["end"] > start:
            optimized[-1]["end"] = max(optimized[-1]["start"] + 0.5, start - 0.02)
            optimized[-1]["end_srt"] = format_seconds_to_srt_timestamp(optimized[-1]["end"])
            optimized[-1]["end_vtt"] = format_seconds_to_vtt_timestamp(optimized[-1]["end"])

        orig_text = seg.get("original_text", "").strip()
        trans_text = seg.get("translated_text", orig_text).strip()
        speaker = seg.get("speaker", "Speaker 1")

        wrapped_trans = _wrap_text(trans_text, max_line_chars)
        wrapped_orig = _wrap_text(orig_text, max_line_chars)

        optimized_seg = {
            "segment_id": len(optimized) + 1,
            "start": round(start, 3),
            "end": round(end, 3),
            "start_srt": format_seconds_to_srt_timestamp(start),
            "end_srt": format_seconds_to_srt_timestamp(end),
            "start_vtt": format_seconds_to_vtt_timestamp(start),
            "end_vtt": format_seconds_to_vtt_timestamp(end),
            "speaker": speaker,
            "original_text": wrapped_orig,
            "translated_text": wrapped_trans
        }
        optimized.append(optimized_seg)

    return optimized


def _wrap_text(text: str, max_chars: int) -> str:
    """Wraps string into max 2 lines for subtitle display."""
    if len(text) <= max_chars:
        return text

    words = text.split()
    line1 = []
    line2 = []
    current_len = 0

    for word in words:
        if current_len + len(word) + 1 <= max_chars and not line2:
            line1.append(word)
            current_len += len(word) + 1
        else:
            line2.append(word)

    if line2:
        return f"{' '.join(line1)}\n{' '.join(line2)}"
    return ' '.join(line1)
