"""
System Performance & Latency Metrics Logger.
Measures STT time, translation time, subtitle generation time, rendering time, and RTF.
Exports structured logs to CSV (translation_log.csv).
"""

import os
import csv
import time
from typing import Dict, Any, List, Optional

CSV_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs", "translation_log.csv")


def measure_execution_time(func, *args, **kwargs):
    """Executes func and returns (result, elapsed_seconds)."""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, round(elapsed, 4)


def compute_real_time_factor(audio_duration_sec: float, processing_time_sec: float) -> float:
    """Computes Real-Time Factor (RTF) = processing_time / audio_duration."""
    if audio_duration_sec <= 0:
        return 0.0
    return round(processing_time_sec / audio_duration_sec, 4)


def export_translation_csv(
    segments: List[Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    model_name: str,
    output_csv_path: Optional[str] = None
) -> str:
    """
    Exports structured subtitle log to CSV with columns:
    segment_id, start_time, end_time, speaker, source_language, target_language, original_text, translated_text, model, processing_time
    """
    target_path = output_csv_path or CSV_LOG_PATH
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    fieldnames = [
        "segment_id",
        "start_time",
        "end_time",
        "speaker",
        "source_language",
        "target_language",
        "original_text",
        "translated_text",
        "model",
        "processing_time"
    ]

    write_header = not os.path.exists(target_path) or os.path.getsize(target_path) == 0

    with open(target_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for seg in segments:
            writer.writerow({
                "segment_id": seg.get("segment_id", 1),
                "start_time": seg.get("start", 0.0),
                "end_time": seg.get("end", 0.0),
                "speaker": seg.get("speaker", "Speaker 1"),
                "source_language": source_lang,
                "target_language": target_lang,
                "original_text": seg.get("original_text", ""),
                "translated_text": seg.get("translated_text", ""),
                "model": model_name,
                "processing_time": seg.get("processing_time", 0.05)
            })

    return target_path
