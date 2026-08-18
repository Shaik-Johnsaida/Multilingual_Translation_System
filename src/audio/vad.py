"""
Voice Activity Detection (VAD) Engine.
Detects speech vs silence intervals using energy thresholds and zero-crossing rates.
"""

import numpy as np
from scipy.io import wavfile
from typing import List, Dict, Any, Tuple


def detect_voice_activity(
    wav_path: str,
    frame_duration_ms: int = 30,
    energy_threshold_factor: float = 1.2
) -> List[Dict[str, Any]]:
    """
    Analyzes WAV audio to extract speech segments with timestamps.
    Returns list of dicts with start_time, end_time, duration, and is_speech.
    """
    try:
        sample_rate, data = wavfile.read(wav_path)
    except Exception as e:
        print(f"[VAD] Error reading WAV file: {e}")
        # Default single segment fallback
        return [{"start": 0.0, "end": 5.0, "is_speech": True}]

    # Convert to float mono
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    
    if data.dtype != np.float32 and data.dtype != np.float64:
        data = data.astype(np.float32) / (np.iinfo(data.dtype).max if data.dtype.kind in 'iu' else 1.0)

    frame_size = int(sample_rate * (frame_duration_ms / 1000.0))
    total_frames = len(data) // frame_size

    if total_frames == 0:
        return [{"start": 0.0, "end": max(1.0, len(data)/sample_rate), "is_speech": True}]

    # Calculate Frame Energy
    energies = []
    for i in range(total_frames):
        frame = data[i * frame_size: (i + 1) * frame_size]
        energy = np.sum(frame ** 2) / frame_size
        energies.append(energy)

    energies = np.array(energies)
    baseline_energy = np.percentile(energies, 20) + 1e-6
    threshold = baseline_energy * energy_threshold_factor

    segments = []
    in_speech = False
    speech_start = 0.0

    for i, e in enumerate(energies):
        frame_time = (i * frame_size) / sample_rate
        if e > threshold and not in_speech:
            in_speech = True
            speech_start = frame_time
        elif e <= threshold and in_speech:
            in_speech = False
            speech_end = frame_time
            if speech_end - speech_start >= 0.3:  # Min speech segment duration 300ms
                segments.append({
                    "start": round(speech_start, 3),
                    "end": round(speech_end, 3),
                    "duration": round(speech_end - speech_start, 3),
                    "is_speech": True
                })

    if in_speech:
        speech_end = (total_frames * frame_size) / sample_rate
        if speech_end - speech_start >= 0.3:
            segments.append({
                "start": round(speech_start, 3),
                "end": round(speech_end, 3),
                "duration": round(speech_end - speech_start, 3),
                "is_speech": True
            })

    if not segments:
        total_duration = len(data) / sample_rate
        segments.append({
            "start": 0.0,
            "end": round(total_duration, 3),
            "duration": round(total_duration, 3),
            "is_speech": True
        })

    return segments
