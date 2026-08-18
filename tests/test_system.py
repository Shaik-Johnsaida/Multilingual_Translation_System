"""
Comprehensive Automated Test Suite for Real-Time Multilingual Audio Dubbing & Translation System.
Tests strictly the 10 Production Supported Languages across Registry, Detector, Translation Engine,
Audio VAD, STT, Diarization, Subtitles, Linear Non-Overlapping TTS Dubbing, Video Overlay, Pipeline, and FastAPI Endpoints.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure root folder is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.language.registry import get_supported_languages, resolve_language_code, get_language_display_name, LANGUAGE_REGISTRY
from src.language.detector import detect_text_language
from src.translation.engine import translation_engine
from src.audio.vad import detect_voice_activity
from src.speech.transcription import stt_engine, get_whisper_lang_code, deduplicate_asr_segments
from src.speech.diarization import assign_speaker_labels
from src.subtitles.segmentation import optimize_subtitle_segments, format_seconds_to_srt_timestamp, format_seconds_to_vtt_timestamp
from src.subtitles.srt import generate_srt_content
from src.subtitles.vtt import generate_vtt_content
from src.audio.tts import synthesize_segment_tts, get_tts_lang_code, TTS_LANG_MAPPING, ConsistentVoiceSynthesizer, generate_dubbed_audio_track
from src.pipeline import pipeline
from src.video.overlay import get_media_duration
from src.evaluation.bleu import compute_bleu_score
from src.evaluation.wer import compute_wer_score
from src.evaluation.latency import compute_real_time_factor
from backend.main import app


# 1. Language Registry Tests (Strictly 10 Languages)
def test_language_registry_strictly_10_languages():
    langs = get_supported_languages()
    assert len(langs) == 10, f"Expected exactly 10 languages, found {len(langs)}"
    
    expected_codes = ["te", "hi", "en", "ur", "ta", "ml", "zh", "ja", "es", "fr"]
    actual_codes = [l["code"] for l in langs]
    assert sorted(actual_codes) == sorted(expected_codes)

    expected_flores = ["tel_Telu", "hin_Deva", "eng_Latn", "urd_Arab", "tam_Taml", "mal_Mlym", "zho_Hans", "jpn_Jpan", "spa_Latn", "fra_Latn"]
    actual_flores = [l["flores_code"] for l in langs]
    assert sorted(actual_flores) == sorted(expected_flores)

    # Resolution checks for the 10 languages
    assert resolve_language_code("Telugu") == "tel_Telu"
    assert resolve_language_code("te") == "tel_Telu"
    assert resolve_language_code("Hindi") == "hin_Deva"
    assert resolve_language_code("hi") == "hin_Deva"
    assert resolve_language_code("English") == "eng_Latn"
    assert resolve_language_code("en") == "eng_Latn"
    assert resolve_language_code("Urdu") == "urd_Arab"
    assert resolve_language_code("ur") == "urd_Arab"
    assert resolve_language_code("Tamil") == "tam_Taml"
    assert resolve_language_code("ta") == "tam_Taml"
    assert resolve_language_code("Malayalam") == "mal_Mlym"
    assert resolve_language_code("ml") == "mal_Mlym"
    assert resolve_language_code("Chinese") == "zho_Hans"
    assert resolve_language_code("zh") == "zho_Hans"
    assert resolve_language_code("Japanese") == "jpn_Jpan"
    assert resolve_language_code("ja") == "jpn_Jpan"
    assert resolve_language_code("Spanish") == "spa_Latn"
    assert resolve_language_code("es") == "spa_Latn"
    assert resolve_language_code("French") == "fra_Latn"
    assert resolve_language_code("fr") == "fra_Latn"

    # Unsupported languages MUST return None (no Marathi, Bengali, German, etc.)
    assert resolve_language_code("Marathi") is None
    assert resolve_language_code("mr") is None
    assert resolve_language_code("Bengali") is None
    assert resolve_language_code("bn") is None
    assert resolve_language_code("German") is None
    assert resolve_language_code("de") is None
    assert resolve_language_code("Arabic") is None
    assert resolve_language_code("ar") is None
    assert resolve_language_code("Russian") is None
    assert resolve_language_code("ru") is None


# 2. Text Language Detector Tests (Strictly within the 10 Supported Languages)
def test_language_detector_supported_10():
    # Telugu
    code_te, _ = detect_text_language("నమస్కారం అందరికీ")
    assert code_te == "tel_Telu"
    
    # Hindi
    code_hi, _ = detect_text_language("नमस्ते आप कैसे हैं")
    assert code_hi == "hin_Deva"

    # English
    code_en, _ = detect_text_language("Hello world and welcome to our system")
    assert code_en == "eng_Latn"

    # Tamil
    code_ta, _ = detect_text_language("வணக்கம் அனைவரும் நலமா")
    assert code_ta == "tam_Taml"

    # Malayalam
    code_ml, _ = detect_text_language("നമസ്കാരം എല്ലാവർക്കും സ്വാഗതം")
    assert code_ml == "mal_Mlym"

    # Urdu
    code_ur, _ = detect_text_language("خوش آمدید آپ کیسے ہیں")
    assert code_ur == "urd_Arab"

    # Japanese
    code_ja, _ = detect_text_language("こんにちは、ようこそ")
    assert code_ja == "jpn_Jpan"

    # Chinese
    code_zh, _ = detect_text_language("你好，欢迎大家来到这里")
    assert code_zh == "zho_Hans"

    # Spanish
    code_es, _ = detect_text_language("¡Hola amigos! Bienvenidos a todos con mucho gusto")
    assert code_es == "spa_Latn"

    # French
    code_fr, _ = detect_text_language("Bonjour à tous et bienvenue à notre présentation")
    assert code_fr == "fra_Latn"

    # Verify that all detected codes exist in LANGUAGE_REGISTRY
    for test_text in ["నమస్కారం", "नमस्ते", "Hello", "வணக்கம்", "നമസ്കാരം", "خوش آمدید", "こんにちは", "你好", "¡Hola!", "Bonjour!"]:
        code, _ = detect_text_language(test_text)
        assert code in LANGUAGE_REGISTRY, f"Detected code {code} is not in the 10 supported languages"


# 3. Universal Translation Engine Tests
def test_translation_engine_pairs():
    # English -> Telugu
    res_te = translation_engine.translate_single("welcome everyone", "eng_Latn", "tel_Telu")
    assert res_te["source_lang"] == "eng_Latn"
    assert res_te["target_lang"] == "tel_Telu"
    assert len(res_te["translated_text"]) > 0

    # English -> Hindi
    res_hi = translation_engine.translate_single("welcome everyone", "eng_Latn", "hin_Deva")
    assert res_hi["target_lang"] == "hin_Deva"
    assert len(res_hi["translated_text"]) > 0

    # English -> French
    res_fr = translation_engine.translate_single("welcome everyone", "eng_Latn", "fra_Latn")
    assert res_fr["target_lang"] == "fra_Latn"
    assert len(res_fr["translated_text"]) > 0

    # English -> Spanish
    res_es = translation_engine.translate_single("welcome everyone", "eng_Latn", "spa_Latn")
    assert res_es["target_lang"] == "spa_Latn"
    assert len(res_es["translated_text"]) > 0

    # English -> Japanese
    res_ja = translation_engine.translate_single("welcome everyone", "eng_Latn", "jpn_Jpan")
    assert res_ja["target_lang"] == "jpn_Jpan"
    assert len(res_ja["translated_text"]) > 0

    # English -> Chinese
    res_zh = translation_engine.translate_single("welcome everyone", "eng_Latn", "zho_Hans")
    assert res_zh["target_lang"] == "zho_Hans"
    assert len(res_zh["translated_text"]) > 0

    # English -> Urdu
    res_ur = translation_engine.translate_single("welcome everyone", "eng_Latn", "urd_Arab")
    assert res_ur["target_lang"] == "urd_Arab"
    assert len(res_ur["translated_text"]) > 0

    # Auto detect test
    res_auto = translation_engine.translate_single("నమస్కారం", "auto", "eng_Latn")
    assert res_auto["detected_source"] == "tel_Telu"
    assert len(res_auto["translated_text"]) > 0

# 4. Single Speaker Diarization — Forced Mode
def test_single_speaker_diarization_consistency():
    """When single_speaker_mode=True, all segments must be labeled 'Speaker 1'."""
    segments = [
        {"start": 0.0, "end": 30.0, "original_text": "First chunk"},
        {"start": 30.0, "end": 60.0, "original_text": "Second chunk"},
        {"start": 60.0, "end": 88.4, "original_text": "Third chunk"}
    ]
    diarized = assign_speaker_labels("dummy.wav", segments, single_speaker_mode=True)
    assert len(diarized) == 3
    for s in diarized:
        assert s["speaker"] == "Speaker 1", f"Expected 'Speaker 1', got {s['speaker']}"


# 4b. Multi-Speaker Diarization — Auto-Detect with Synthetic Audio
def test_multi_speaker_diarization_auto_detect():
    """
    Creates two synthetic WAV audio signals with very different spectral characteristics
    (low-frequency tone vs high-frequency tone) to simulate 2 distinct speakers.
    The diarization engine must assign them to different speaker IDs.
    """
    import wave, struct, math, tempfile

    sample_rate = 16000
    duration_per_seg = 2.0  # seconds per segment
    n_samples = int(sample_rate * duration_per_seg)

    # Speaker A: 200 Hz tone (low male-like)
    tone_a = [int(16000 * math.sin(2 * math.pi * 200 * i / sample_rate)) for i in range(n_samples)]
    # Speaker B: 1600 Hz tone (high female-like)
    tone_b = [int(16000 * math.sin(2 * math.pi * 1600 * i / sample_rate)) for i in range(n_samples)]

    # Build a WAV with: A, B, A, B (4 segments, 2 speakers alternating)
    combined = tone_a + tone_b + tone_a + tone_b
    total_samples = len(combined)

    tmp_wav = tempfile.mktemp(suffix=".wav")
    with wave.open(tmp_wav, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"{total_samples}h", *combined))

    seg_dur = duration_per_seg
    segments = [
        {"start": 0.0,         "end": seg_dur,       "original_text": "Segment A1"},   # Speaker A
        {"start": seg_dur,     "end": seg_dur * 2,   "original_text": "Segment B1"},   # Speaker B
        {"start": seg_dur * 2, "end": seg_dur * 3,   "original_text": "Segment A2"},   # Speaker A
        {"start": seg_dur * 3, "end": seg_dur * 4,   "original_text": "Segment B2"},   # Speaker B
    ]

    diarized = assign_speaker_labels(tmp_wav, segments, single_speaker_mode=False)

    import os
    try:
        os.remove(tmp_wav)
    except Exception:
        pass

    assert len(diarized) == 4
    unique_speakers = set(s["speaker"] for s in diarized)
    # With distinctly different tones, must detect 2 speakers
    assert len(unique_speakers) == 2, f"Expected 2 speakers, got {len(unique_speakers)}: {unique_speakers}"
    # First and third segment (A) must share the same speaker
    assert diarized[0]["speaker"] == diarized[2]["speaker"], "Speaker A must have consistent ID"
    # Second and fourth segment (B) must share the same speaker
    assert diarized[1]["speaker"] == diarized[3]["speaker"], "Speaker B must have consistent ID"
    # A and B must be different
    assert diarized[0]["speaker"] != diarized[1]["speaker"], "Speaker A and Speaker B must be different"

# 5. ASR Deduplication & Clamping Test
def test_asr_deduplication_and_clamping():
    raw_segments = [
        {"start": 0.0, "end": 5.0, "text": "Hello world"},
        {"start": 5.0, "end": 10.0, "text": "Hello world"},  # Duplicate repeating segment
        {"start": 10.0, "end": 15.0, "text": "Next unique sentence"},
        {"start": 90.0, "end": 95.0, "text": "Out of bounds sentence"}  # Past max_duration
    ]
    cleaned = deduplicate_asr_segments(raw_segments, max_duration=88.7)
    assert len(cleaned) == 2
    assert cleaned[0]["original_text"] == "Hello world"
    assert cleaned[0]["end"] == 10.0  # Merged duration
    assert cleaned[1]["original_text"] == "Next unique sentence"


# 6. Text-to-Speech (TTS) Dubbing Codes & Consistency Test
def test_tts_dubbing_codes_strictly_10():
    assert len(TTS_LANG_MAPPING) == 10
    assert get_tts_lang_code("tel_Telu") == "te"
    assert get_tts_lang_code("hin_Deva") == "hi"
    assert get_tts_lang_code("eng_Latn") == "en"
    assert get_tts_lang_code("urd_Arab") == "ur"
    assert get_tts_lang_code("tam_Taml") == "ta"
    assert get_tts_lang_code("mal_Mlym") == "ml"
    assert get_tts_lang_code("zho_Hans") == "zh-CN"
    assert get_tts_lang_code("jpn_Jpan") == "ja"
    assert get_tts_lang_code("spa_Latn") == "es"
    assert get_tts_lang_code("fra_Latn") == "fr"

    synth = ConsistentVoiceSynthesizer("tel_Telu")
    assert synth.lang_code == "te"


# 7. Non-Overlapping Audio Dubbing Timeline Test
def test_tts_non_overlapping_audio_dubbing():
    test_subtitles = [
        {"start": 0.0, "end": 4.0, "translated_text": "Hello world", "speaker": "Speaker 1"},
        {"start": 4.0, "end": 8.0, "translated_text": "Welcome to our system", "speaker": "Speaker 1"},
        {"start": 8.0, "end": 12.0, "translated_text": "This is a non-overlapping audio test", "speaker": "Speaker 1"}
    ]
    out_wav = os.path.join(ROOT_DIR, "outputs", "test_dubbing_track.wav")
    success, res_info = generate_dubbed_audio_track(test_subtitles, "eng_Latn", 12.0, out_wav)
    assert success is True
    assert res_info["overlapping_segments_count"] == 0, f"Expected 0 overlapping segments, got {res_info['overlapping_segments_count']}"
    assert res_info["voice_count"] == 1
    assert os.path.exists(out_wav)


# 8. Subtitle Timestamp & Format Tests
def test_subtitle_timestamps():
    assert format_seconds_to_srt_timestamp(3.5) == "00:00:03,500"
    assert format_seconds_to_vtt_timestamp(3.5) == "00:00:03.500"

    raw_segments = [
        {"start": 1.0, "end": 4.0, "original_text": "Hello world", "translated_text": "నమస్కారం ప్రపంచం", "speaker": "Speaker 1"}
    ]
    opt = optimize_subtitle_segments(raw_segments, max_duration=10.0)
    assert len(opt) == 1
    assert opt[0]["start_srt"] == "00:00:01,000"

    srt_str = generate_srt_content(opt)
    assert "00:00:01,000 --> 00:00:04,000" in srt_str

    vtt_str = generate_vtt_content(opt)
    assert "WEBVTT" in vtt_str


# 9. Evaluation Metrics Tests
def test_evaluation_metrics():
    bleu_res = compute_bleu_score("This is a test translation", "This is a test translation")
    assert bleu_res["bleu"] == 100.0

    wer_res = compute_wer_score("hello world", "hello world")
    assert wer_res["wer"] == 0.0


# 10. Pipeline Text Mode Test
def test_pipeline_text_mode():
    res = pipeline.process_text("Welcome to our multilingual system", "eng_Latn", "tel_Telu")
    assert res["target_lang"] == "tel_Telu"
    assert len(res["translated_text"]) > 0


# 11. FastAPI Endpoint Tests
def test_fastapi_endpoints():
    client = TestClient(app)

    # Root redirect
    r_root = client.get("/", follow_redirects=False)
    assert r_root.status_code in (302, 307)
    assert "/dashboard/index.html" in r_root.headers["location"]

    # Languages endpoint must return exactly 10 languages
    r_lang = client.get("/api/languages")
    assert r_lang.status_code == 200
    data = r_lang.json()
    assert data["total_count"] == 10
    assert len(data["languages"]) == 10
    codes = [l["code"] for l in data["languages"]]
    assert sorted(codes) == ["en", "es", "fr", "hi", "ja", "ml", "ta", "te", "ur", "zh"]

    # Status endpoint
    r_stat = client.get("/api/status")
    assert r_stat.status_code == 200
    assert r_stat.json()["languages_count"] == 10

    # Text translation endpoint (English -> Telugu)
    r_trans = client.post("/api/translate/text", json={
        "text": "welcome everyone",
        "source_lang": "eng_Latn",
        "target_lang": "tel_Telu"
    })
    assert r_trans.status_code == 200
    assert r_trans.json()["status"] == "success"
    assert len(r_trans.json()["result"]["translated_text"]) > 0

    # History endpoint
    r_hist = client.get("/api/history")
    assert r_hist.status_code == 200
    assert "history" in r_hist.json()
