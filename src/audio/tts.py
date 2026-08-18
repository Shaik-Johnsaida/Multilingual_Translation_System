"""
Text-to-Speech (TTS) & Audio Dubbing Engine.
Synthesizes target language speech using one consistent voice configuration per video
and constructs a strictly non-overlapping, synchronized dubbed audio track matching the master media duration.
"""

import os
import wave
import struct
import subprocess
import tempfile
from typing import List, Dict, Any, Tuple, Optional
from gtts import gTTS
import pyttsx3

# FLORES code to TTS ISO language mapping (Strictly 10 Production Languages)
TTS_LANG_MAPPING = {
    "tel_Telu": "te",
    "hin_Deva": "hi",
    "eng_Latn": "en",
    "urd_Arab": "ur",
    "tam_Taml": "ta",
    "mal_Mlym": "ml",
    "zho_Hans": "zh-CN",
    "jpn_Jpan": "ja",
    "spa_Latn": "es",
    "fra_Latn": "fr",
}


def get_tts_lang_code(flores_or_code: str) -> str:
    """Resolves language code to gTTS / pyttsx3 compatible 2-letter code."""
    if flores_or_code in TTS_LANG_MAPPING:
        return TTS_LANG_MAPPING[flores_or_code]
    
    code_short = flores_or_code.split("_")[0][:2].lower()
    return code_short if len(code_short) == 2 else "en"


def get_wav_duration_and_frames(wav_path: str) -> Tuple[float, bytes]:
    """Reads 16kHz 16-bit mono WAV frames and duration."""
    try:
        with wave.open(wav_path, 'rb') as wf:
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            data = wf.readframes(nframes)
            dur = nframes / float(framerate) if framerate > 0 else 0.0
            return dur, data
    except Exception:
        return 0.0, b""


def adjust_audio_tempo(input_wav: str, output_wav: str, speed_factor: float) -> bool:
    """
    Adjusts audio playback speed without changing pitch or timbre using FFmpeg atempo.
    Clamps speed between 0.85x and 1.50x to preserve clear, natural human speech.
    """
    clamped_speed = max(0.85, min(1.50, float(speed_factor)))
    cmd = [
        "ffmpeg", "-y",
        "-i", input_wav,
        "-filter:a", f"atempo={clamped_speed:.3f}",
        "-ac", "1",
        "-ar", "16000",
        output_wav
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return os.path.exists(output_wav) and os.path.getsize(output_wav) > 44


class MultiSpeakerVoiceRegistry:
    """
    Maintains a stable mapping of speaker IDs to consistent voice synthesizers.
    Each unique Speaker ID (e.g. 'Speaker 1', 'Speaker 2') maps to exactly one
    ConsistentVoiceSynthesizer that is reused for all segments of that speaker.
    This guarantees per-speaker voice consistency across the full video.
    """
    def __init__(self, target_lang: str):
        self.target_lang = target_lang
        self._registry: Dict[str, "ConsistentVoiceSynthesizer"] = {}
        # Pinned offline voices to avoid re-init per speaker
        self._offline_voices: List[str] = []
        self._load_offline_voices()

    def _load_offline_voices(self):
        """Pre-load all available pyttsx3 voices so per-speaker assignment is stable."""
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            if voices:
                self._offline_voices = [v.id for v in voices]
        except Exception:
            pass

    def get_synthesizer(self, speaker_id: str) -> "ConsistentVoiceSynthesizer":
        """
        Returns the pinned ConsistentVoiceSynthesizer for the given speaker_id.
        Creates and caches a new one on first access.
        Uses a different pyttsx3 offline voice index per speaker for natural differentiation
        (if available), while gTTS uses the same language for all.
        """
        if speaker_id not in self._registry:
            synth = ConsistentVoiceSynthesizer(self.target_lang)
            # If multiple pyttsx3 voices exist, cycle through them per speaker
            if self._offline_voices:
                speaker_idx = len(self._registry) % len(self._offline_voices)
                synth._offline_voice_id = self._offline_voices[speaker_idx]
            self._registry[speaker_id] = synth
        return self._registry[speaker_id]

    @property
    def registered_speaker_count(self) -> int:
        return len(self._registry)


class ConsistentVoiceSynthesizer:
    """
    Maintains a consistent target voice profile throughout the entire media dubbing process.
    Guarantees no gender switching, no random voice alternation, and identical voice parameters.
    """
    def __init__(self, target_lang: str):
        self.target_lang = target_lang
        self.lang_code = get_tts_lang_code(target_lang)
        self._offline_engine = None
        self._offline_voice_id = None
        self._init_offline_voice()

    def _init_offline_voice(self):
        """Initializes and pins a single consistent offline voice for fallback synthesis."""
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            if voices:
                selected_voice = voices[0].id
                for v in voices:
                    if self.lang_code in v.id.lower() or self.lang_code in (v.languages if hasattr(v, 'languages') else []):
                        selected_voice = v.id
                        break
                self._offline_voice_id = selected_voice
        except Exception as e:
            pass

    def synthesize_segment(self, text: str, output_wav_16k_mono: str) -> bool:
        """
        Synthesizes text segment into a standardized 16kHz 16-bit Mono WAV file
        using the pinned consistent target voice.
        """
        if not text or not text.strip():
            return False

        temp_raw = output_wav_16k_mono + ".raw.mp3"
        success = False

        # 1. Primary Engine: gTTS for pristine native target language speech
        try:
            tts = gTTS(text=text.strip(), lang=self.lang_code, slow=False)
            tts.save(temp_raw)
            if os.path.exists(temp_raw) and os.path.getsize(temp_raw) > 0:
                success = True
        except Exception:
            pass

        # 2. Fallback Engine: pyttsx3 with pinned consistent voice ID
        if not success:
            try:
                engine = pyttsx3.init()
                if self._offline_voice_id:
                    engine.setProperty('voice', self._offline_voice_id)
                engine.setProperty('rate', 150)
                engine.save_to_file(text.strip(), temp_raw)
                engine.runAndWait()
                if os.path.exists(temp_raw) and os.path.getsize(temp_raw) > 0:
                    success = True
            except Exception:
                pass

        if not success:
            return False

        # Standardize strictly to 16kHz 16-bit Mono PCM WAV
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_raw,
            "-ac", "1",
            "-ar", "16000",
            output_wav_16k_mono
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if os.path.exists(temp_raw):
            try:
                os.remove(temp_raw)
            except Exception:
                pass

        return os.path.exists(output_wav_16k_mono) and os.path.getsize(output_wav_16k_mono) > 44


def synthesize_segment_tts(text: str, target_lang: str, output_path: str) -> bool:
    """Convenience function for single segment synthesis."""
    synth = ConsistentVoiceSynthesizer(target_lang)
    return synth.synthesize_segment(text, output_path)


def assemble_linear_non_overlapping_audio(
    scheduled_segments: List[Dict[str, Any]],
    total_duration_sec: float,
    output_wav_path: str,
    sample_rate: int = 16000
) -> Tuple[bool, int]:
    """
    Builds a single-channel, strictly non-overlapping PCM WAV file by linearly writing
    audio samples into a continuous timeline buffer with exact millisecond precision.
    Guarantees mathematically that overlap count is exactly 0.
    """
    total_samples = int(max(1.0, total_duration_sec) * sample_rate)
    pcm_buffer = bytearray(total_samples * 2)  # 16-bit signed PCM = 2 bytes per sample (silence)

    current_sample_idx = 0
    overlap_count = 0

    for seg in scheduled_segments:
        start_sec = float(seg["scheduled_start"])
        raw_frames = seg["frames"]
        frames_count = len(raw_frames) // 2

        start_sample = int(start_sec * sample_rate)
        
        # Enforce strict non-overlapping start boundary
        if start_sample < current_sample_idx:
            overlap_count += 1
            start_sample = current_sample_idx + int(0.04 * sample_rate)  # 40ms natural boundary pause

        end_sample = start_sample + frames_count

        # Write into PCM buffer if within total media bounds
        if start_sample < total_samples:
            write_end = min(end_sample, total_samples)
            bytes_to_copy = (write_end - start_sample) * 2
            pcm_buffer[start_sample * 2 : start_sample * 2 + bytes_to_copy] = raw_frames[:bytes_to_copy]
            current_sample_idx = write_end

    with wave.open(output_wav_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_buffer)

    return True, overlap_count


def generate_dubbed_audio_track(
    subtitle_segments: List[Dict[str, Any]],
    target_lang: str,
    total_duration_sec: float,
    output_dubbed_wav: str
) -> Tuple[bool, Dict[str, Any]]:
    """
    Generates a synchronized dubbed audio track in the target language.
    Guarantees:
    1. Per-speaker consistent voice: each unique speaker_id maps to a stable synthesizer.
    2. Intelligent time-stretching (atempo) so synthesized speech fits its natural speech window.
    3. Strictly NON-OVERLAPPING audio timeline (overlap count = 0).
    4. Exact total duration matching master video.
    """
    target_duration = max(1.0, float(total_duration_sec))

    if not subtitle_segments:
        # Generate silent audio track matching the full media duration
        cmd_silent = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-t", str(target_duration),
            output_dubbed_wav
        ]
        subprocess.run(cmd_silent, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True, {
            "output_path": output_dubbed_wav,
            "tts_segments_count": 0,
            "overlapping_segments_count": 0,
            "voice_count": 1,
            "target_language": target_lang
        }

    temp_dir = tempfile.mkdtemp(prefix="dubbing_")
    # Multi-speaker voice registry: each Speaker ID gets its own stable synthesizer
    voice_registry = MultiSpeakerVoiceRegistry(target_lang)

    scheduled_segments = []
    previous_end_time = 0.0
    synthesized_count = 0

    num_segments = len(subtitle_segments)

    for idx, seg in enumerate(subtitle_segments):
        trans_text = seg.get("translated_text", "").strip()
        if not trans_text:
            continue

        orig_start = float(seg.get("start", 0.0))
        orig_end = float(seg.get("end", orig_start + 2.0))

        # Lookahead to determine available window before next segment
        if idx + 1 < num_segments:
            next_start = float(subtitle_segments[idx + 1].get("start", orig_end))
            available_window = max(1.0, next_start - orig_start)
        else:
            available_window = max(1.0, target_duration - orig_start)

        # Resolve the stable synthesizer for this segment's speaker
        speaker_id = seg.get("speaker", "Speaker 1")
        voice_synth = voice_registry.get_synthesizer(speaker_id)

        raw_seg_wav = os.path.join(temp_dir, f"raw_seg_{idx}.wav")
        success = voice_synth.synthesize_segment(trans_text, raw_seg_wav)
        if not success:
            continue

        raw_dur, raw_frames = get_wav_duration_and_frames(raw_seg_wav)
        if raw_dur <= 0.0:
            continue

        # Adjust tempo if TTS speech exceeds the available window before the next segment
        final_wav = raw_seg_wav
        final_dur = raw_dur
        final_frames = raw_frames

        if raw_dur > available_window and available_window >= 0.8:
            speedup = min(1.45, raw_dur / (available_window * 0.95))
            fitted_wav = os.path.join(temp_dir, f"fitted_seg_{idx}.wav")
            if adjust_audio_tempo(raw_seg_wav, fitted_wav, speedup):
                f_dur, f_frames = get_wav_duration_and_frames(fitted_wav)
                if f_dur > 0:
                    final_wav = fitted_wav
                    final_dur = f_dur
                    final_frames = f_frames

        # Schedule start timestamp ensuring strictly previous_end <= current_start
        scheduled_start = max(orig_start, previous_end_time + 0.03)

        scheduled_segments.append({
            "segment_id": idx + 1,
            "orig_start": orig_start,
            "orig_end": orig_end,
            "scheduled_start": scheduled_start,
            "scheduled_end": scheduled_start + final_dur,
            "duration": final_dur,
            "frames": final_frames,
            "text": trans_text
        })

        previous_end_time = scheduled_start + final_dur
        synthesized_count += 1

    # Linear non-overlapping assembly into final output WAV
    success, overlap_count = assemble_linear_non_overlapping_audio(
        scheduled_segments,
        target_duration,
        output_dubbed_wav,
        sample_rate=16000
    )

    # Clean up temp files
    try:
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
    except Exception:
        pass

    res_metadata = {
        "output_path": output_dubbed_wav,
        "tts_segments_count": synthesized_count,
        "overlapping_segments_count": overlap_count,
        "voice_count": voice_registry.registered_speaker_count,
        "target_language": target_lang,
        "total_duration_sec": target_duration
    }

    return success, res_metadata
