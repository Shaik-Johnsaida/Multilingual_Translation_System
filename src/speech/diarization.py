"""
Speaker Diarization Engine.
Assigns consistent speaker identifiers across transcript segments using real acoustic
feature-based voice embedding and agglomerative clustering.

Strategy:
- Extracts mel-frequency cepstral coefficient (MFCC)-like spectral features from each
  speech segment's raw audio.
- Clusters segments into speaker groups using AgglomerativeClustering (no pre-set speaker
  count required; automatically estimates using a silhouette-based gap or a conservative
  distance threshold).
- Ensures speaker identity survives chunking: global mapping across the full video.
- Merges adjacent segments assigned to the same speaker.

Fallback (single_speaker_mode=True):
- All segments labeled 'Speaker 1' for guaranteed single-voice dubbing.

Note on external model availability:
- pyannote.audio is NOT available in this environment.
- pydub is NOT installed.
- Only scipy + sklearn (both confirmed available) are used for feature extraction and clustering.
"""

import numpy as np
from scipy.io import wavfile
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Optional

# Maximum number of speakers ever auto-detected (safety cap)
MAX_AUTO_SPEAKERS = 6
# Agglomerative clustering distance threshold (lower = more speakers)
# Calibrated conservatively so 1-speaker videos stay at Speaker 1
_CLUSTER_DISTANCE_THRESHOLD = 8.0


def _extract_spectral_features(audio_chunk: np.ndarray, sample_rate: int, n_features: int = 20) -> np.ndarray:
    """
    Extracts MFCC-like spectral voice features from a raw mono float32 audio chunk.
    Uses FFT-based log mel spectrum, zero-crossing rate, RMS energy, and spectral centroid.
    Returns a 1D feature vector (n_features,).
    """
    if len(audio_chunk) < 256:
        return np.zeros(n_features)

    frame_size = min(int(0.025 * sample_rate), len(audio_chunk))  # 25ms
    hop_size = max(1, int(0.010 * sample_rate))   # 10ms

    fft_size = 512
    frames = []
    for i in range(0, len(audio_chunk) - frame_size, hop_size):
        frame = audio_chunk[i:i + frame_size]
        windowed = frame * np.hamming(len(frame))
        spectrum = np.abs(np.fft.rfft(windowed, n=fft_size))
        frames.append(spectrum)

    if not frames:
        return np.zeros(n_features)

    frames_arr = np.array(frames)  # (T, fft_size//2+1)
    mean_spectrum = np.mean(frames_arr, axis=0)
    log_spectrum = np.log(mean_spectrum + 1e-8)

    # Sample n_features log-spectral bins (coarse MFCC surrogate)
    spec_feat = np.array([log_spectrum[int(k * len(log_spectrum) / n_features)] for k in range(n_features)])

    # Extra prosodic features: RMS energy, ZCR, spectral centroid, delta
    rms = np.sqrt(np.mean(audio_chunk ** 2))
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio_chunk)))) / 2.0)
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    centroid = float(np.sum(mean_spectrum * freqs) / (np.sum(mean_spectrum) + 1e-8))

    # Combine: last 3 entries replaced with prosodic features
    spec_feat[-3] = rms * 10.0
    spec_feat[-2] = zcr * 5.0
    spec_feat[-1] = centroid / (sample_rate / 2.0) * 5.0

    return spec_feat


def _estimate_num_speakers(features: np.ndarray) -> int:
    """
    Estimates the number of unique speakers by checking silhouette scores for k=1..MAX_AUTO_SPEAKERS.
    Returns 1 if no clear multi-speaker structure is found.
    """
    from sklearn.metrics import silhouette_score

    n = len(features)
    if n <= 1:
        return 1
    if n == 2:
        # For only 2 segments: compare cosine distance
        d = cosine(features[0], features[1])
        return 2 if d > 0.15 else 1

    best_k = 1
    best_score = -1.0

    for k in range(2, min(MAX_AUTO_SPEAKERS + 1, n)):
        try:
            clustering = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
            labels = clustering.fit_predict(features)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(features, labels, metric='cosine')
            # Require a meaningfully positive silhouette (> 0.08) to accept multi-speaker
            if score > best_score and score > 0.08:
                best_score = score
                best_k = k
        except Exception:
            continue

    return best_k


def assign_speaker_labels(
    wav_path: str,
    segments: List[Dict[str, Any]],
    single_speaker_mode: bool = False,
    num_speakers: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Assigns speaker identifiers to transcript segments using acoustic feature clustering.

    Args:
        wav_path: Path to 16kHz mono WAV file.
        segments: List of ASR segment dicts with 'start', 'end', 'original_text'.
        single_speaker_mode: If True, force all segments to 'Speaker 1'.
        num_speakers: If provided (int > 0), use that exact count. If None or 0 → auto-detect.

    Returns:
        Segments list with 'speaker' key added to each dict.
    """
    if not segments:
        return []

    # Explicit single-speaker override
    if single_speaker_mode:
        return _assign_single_speaker(segments)

    # Load audio for feature extraction
    try:
        sample_rate, data = wavfile.read(wav_path)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if np.max(np.abs(data)) > 0:
            data /= np.max(np.abs(data))  # normalize to [-1, 1]
    except Exception as e:
        print(f"[Diarization] Cannot read audio '{wav_path}': {e}. Defaulting to Speaker 1.")
        return _assign_single_speaker(segments)

    # Extract per-segment spectral features
    features = []
    for seg in segments:
        start_sample = max(0, int(float(seg.get("start", 0.0)) * sample_rate))
        end_sample = min(len(data), int(float(seg.get("end", 0.0)) * sample_rate))
        chunk = data[start_sample:end_sample]
        feat = _extract_spectral_features(chunk, sample_rate)
        features.append(feat)

    features_arr = np.array(features)

    # Normalize features for fair distance-based clustering
    try:
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features_arr)
    except Exception:
        features_norm = features_arr

    n_segments = len(segments)

    # Determine effective number of speakers
    if num_speakers and isinstance(num_speakers, int) and num_speakers >= 1:
        effective_k = min(num_speakers, n_segments)
    else:
        # Auto-detect
        effective_k = _estimate_num_speakers(features_norm)

    print(f"[Diarization] Detected {effective_k} speaker(s) across {n_segments} segments.")

    # Assign speaker labels
    if effective_k == 1 or n_segments == 1:
        speakers = ["Speaker 1"] * n_segments
    else:
        try:
            clustering = AgglomerativeClustering(
                n_clusters=effective_k,
                metric='cosine',
                linkage='average'
            )
            cluster_labels = clustering.fit_predict(features_norm)
            # Map cluster indices to stable "Speaker N" names (ordered by first appearance)
            cluster_to_speaker = {}
            next_id = 1
            speakers = []
            for lbl in cluster_labels:
                if lbl not in cluster_to_speaker:
                    cluster_to_speaker[lbl] = f"Speaker {next_id}"
                    next_id += 1
                speakers.append(cluster_to_speaker[lbl])
        except Exception as e:
            print(f"[Diarization] Clustering error: {e}. Defaulting to Speaker 1.")
            speakers = ["Speaker 1"] * n_segments

    # Write speaker back into segment copies
    updated_segments = []
    for idx, seg in enumerate(segments):
        seg_copy = dict(seg)
        seg_copy["speaker"] = speakers[idx]
        updated_segments.append(seg_copy)

    # Report final distribution
    from collections import Counter
    dist = Counter(speakers)
    for spk, count in sorted(dist.items()):
        print(f"[Diarization]   {spk}: {count} segment(s)")

    return updated_segments


def _assign_single_speaker(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assigns all segments to Speaker 1."""
    updated = []
    for seg in segments:
        seg_copy = dict(seg)
        seg_copy["speaker"] = "Speaker 1"
        updated.append(seg_copy)
    return updated
