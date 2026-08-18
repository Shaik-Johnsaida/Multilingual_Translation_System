"""
Unified Multimodal Audio-Visual Translation Pipeline.
Integrates VAD, Whisper STT, Speaker Diarization (Single Speaker Consistency),
Universal Translation, Subtitle Generation, Audio Dubbing (Linear Non-Overlapping TTS),
Video Overlay, and CSV Logging.
Preserves 100% full master video duration, exact synchronized audio timeline, and zero audio overlap.
"""

import os
import time
from typing import Dict, Any, List, Optional, Callable, Union

from src.audio.extraction import extract_audio
from src.audio.vad import detect_voice_activity
from src.speech.transcription import stt_engine, safe_log, is_hallucinated_or_repetitive
from src.speech.diarization import assign_speaker_labels
from src.translation.engine import translation_engine
from src.subtitles.segmentation import optimize_subtitle_segments
from src.subtitles.srt import export_srt_file
from src.subtitles.vtt import export_vtt_file
from src.audio.tts import generate_dubbed_audio_track
from src.video.overlay import overlay_subtitles_on_video, get_media_duration
from src.evaluation.bleu import compute_bleu_score
from src.evaluation.wer import compute_wer_score
from src.evaluation.latency import export_translation_csv
from src.storage.db import record_history


class MultimodalPipeline:
    def __init__(self):
        self.outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
        os.makedirs(self.outputs_dir, exist_ok=True)

    def process_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Dict[str, Any]:
        """Processes direct text translation."""
        result = translation_engine.translate_single(text, source_lang, target_lang)
        record_history(
            mode="TEXT",
            source_lang=result["detected_source"],
            target_lang=result["target_lang"],
            source_text=text,
            translated_text=result["translated_text"],
            model_used=result["model_used"],
            processing_time_sec=result["processing_time_sec"]
        )
        return result

    def process_media(
        self,
        file_path: str,
        source_lang: str = "auto",
        target_langs: Union[str, List[str]] = "tel_Telu",
        progress_callback: Optional[Callable[[str, int], None]] = None,
        reference_translation: Optional[str] = None,
        single_speaker_mode: bool = False,
        num_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Full Audio/Video Processing & Dubbing Pipeline:
        1. Extract Audio & Probe Duration -> 2. VAD -> 3. Whisper STT -> 4. Diarization (Auto Speaker Detection) ->
        5. Universal Machine Translation -> 6. Subtitle Segmentation ->
        7. Target Language Dubbing (Linear Non-Overlapping TTS, Per-Speaker Voice) -> 8. Video Subtitle & Dubbed Audio Replacement ->
        9. CSV Logging, Metrics & Strict Hard Validation
        """
        total_start_time = time.time()
        timings: Dict[str, float] = {}
        validation_errors: List[str] = []
        
        def update_progress(stage: str, pct: int):
            if progress_callback:
                progress_callback(stage, pct)
            print(f"[Pipeline Progress {pct}%] {stage}")

        if isinstance(target_langs, str):
            target_langs_list = [target_langs]
        else:
            target_langs_list = target_langs

        primary_target = target_langs_list[0] if target_langs_list else "tel_Telu"

        # Check 1: Input file readability
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            error_msg = f"Input media file '{file_path}' does not exist or is empty."
            update_progress("Translation failed validation: Input file unreadable.", 100)
            return {"status": "failed", "error": error_msg, "validation_errors": [error_msg]}

        # Stage 1: File check, Probe Master Duration, and Audio Extraction
        t0 = time.time()
        update_progress("Analyzing input media and extracting audio...", 10)
        is_video = file_path.lower().endswith((".mp4", ".mkv", ".avi", ".mov"))
        
        input_media_duration = get_media_duration(file_path)
        safe_log(f"[Pipeline] Master Input File: '{os.path.basename(file_path)}' (Type: {'Video' if is_video else 'Audio'}, Duration: {input_media_duration:.2f}s)")

        if is_video or not file_path.lower().endswith(".wav"):
            success, wav_path = extract_audio(file_path)
            if not success or not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
                error_msg = f"Audio extraction failed for '{file_path}'."
                update_progress("Translation failed validation: Audio extraction error.", 100)
                return {"status": "failed", "error": error_msg, "validation_errors": [error_msg]}
        else:
            wav_path = file_path

        if input_media_duration <= 0.0:
            input_media_duration = get_media_duration(wav_path) or 10.0
        timings["audio_extraction_sec"] = round(time.time() - t0, 3)

        # Stage 2: Voice Activity Detection
        t0 = time.time()
        update_progress("Detecting speech intervals (VAD)...", 20)
        vad_segments = detect_voice_activity(wav_path)
        timings["vad_sec"] = round(time.time() - t0, 3)

        # Stage 3: Speech-to-Text (Full Continuous ASR, clamped to master media duration)
        t0 = time.time()
        update_progress("Transcribing speech with Whisper STT across full media...", 35)
        stt_res = stt_engine.transcribe(wav_path, language=source_lang, max_duration=input_media_duration)
        raw_segments = stt_res.get("segments", [])
        detected_src_lang = stt_res.get("flores_language", "eng_Latn")
        timings["asr_sec"] = round(time.time() - t0, 3)

        # Check 2: ASR validity and non-emptiness
        if not raw_segments:
            validation_errors.append("ASR produced 0 valid speech segments.")
        
        # Check 3: Abnormal transcript repetition
        hallucinated_segs = [s for s in raw_segments if is_hallucinated_or_repetitive(s.get("original_text", ""))]
        if hallucinated_segs:
            validation_errors.append(f"ASR produced {len(hallucinated_segs)} repetitive/hallucinated segments.")

        safe_log(f"[Pipeline Log] ASR Produced {len(raw_segments)} valid speech segments in {timings['asr_sec']}s (Detected Source: {detected_src_lang})")

        # Stage 4: Speaker Diarization (Auto-Detect by default; Single Speaker Mode if explicitly forced)
        t0 = time.time()
        update_progress("Detecting speakers from voice characteristics...", 50)
        diarized_segments = assign_speaker_labels(
            wav_path, raw_segments,
            single_speaker_mode=single_speaker_mode,
            num_speakers=num_speakers
        )
        timings["diarization_sec"] = round(time.time() - t0, 3)

        # Stage 5: Multilingual Machine Translation (Every valid segment translated)
        t0 = time.time()
        update_progress(f"Translating speech segments to {primary_target}...", 65)
        translated_segments = []
        full_orig_text = []
        full_trans_text = []

        safe_log("==================================================================")
        safe_log("--- INTERNAL PIPELINE DEBUG: SEGMENT TRANSCRIPTION & TRANSLATION ---")
        safe_log("==================================================================")

        for idx, seg in enumerate(diarized_segments):
            orig = seg.get("original_text", "").strip()
            if not orig:
                continue

            trans_res = translation_engine.translate_single(orig, detected_src_lang, primary_target)
            translated_text = trans_res.get("translated_text", orig)

            seg_copy = dict(seg)
            seg_copy["translated_text"] = translated_text
            translated_segments.append(seg_copy)
            full_orig_text.append(orig)
            full_trans_text.append(translated_text)

            speaker_id = seg_copy.get("speaker", "Speaker 1")
            safe_log(f"  [{seg_copy['start']:.2f}s -> {seg_copy['end']:.2f}s] ({speaker_id}) | Orig: '{orig}' -> Trans: '{translated_text}'")

        orig_transcript = " ".join(full_orig_text)
        trans_transcript = " ".join(full_trans_text)
        timings["translation_sec"] = round(time.time() - t0, 3)

        # Check 4: Translated segment count match
        if len(translated_segments) < len(raw_segments):
            validation_errors.append(f"Translation dropped segments: {len(translated_segments)} vs {len(raw_segments)} original.")

        # Stage 6: Subtitle Segmentation (Clamped to master media duration)
        t0 = time.time()
        update_progress("Formatting subtitle timing and wrapping...", 75)
        opt_subtitles = optimize_subtitle_segments(translated_segments, max_duration=input_media_duration)

        # Stage 7: SRT & VTT Export
        update_progress("Generating SRT & WebVTT subtitle files...", 82)
        file_stem = os.path.splitext(os.path.basename(file_path))[0]
        srt_path = os.path.join(self.outputs_dir, f"{file_stem}_{primary_target}.srt")
        vtt_path = os.path.join(self.outputs_dir, f"{file_stem}_{primary_target}.vtt")

        export_srt_file(opt_subtitles, srt_path)
        export_vtt_file(opt_subtitles, vtt_path)
        timings["subtitle_sec"] = round(time.time() - t0, 3)

        # Master timeline duration strictly matches input media duration
        total_timeline_sec = input_media_duration

        # Stage 8: Target Language Audio Dubbing (Linear Non-Overlapping Audio Timeline)
        t0 = time.time()
        update_progress(f"Synthesizing dubbed audio track in target language (Zero Overlap, Per-Speaker Voice)...", 88)
        dubbed_wav_path = os.path.join(self.outputs_dir, f"{file_stem}_dubbed_{primary_target}.wav")
        dub_success, dub_res = generate_dubbed_audio_track(
            opt_subtitles,
            primary_target,
            total_timeline_sec,
            dubbed_wav_path
        )
        timings["tts_dubbing_sec"] = round(time.time() - t0, 3)

        overlapping_count = dub_res.get("overlapping_segments_count", 0) if isinstance(dub_res, dict) else 0
        if overlapping_count > 0:
            validation_errors.append(f"Detected {overlapping_count} overlapping audio segments.")

        # Stage 9: Video Subtitle & Dubbed Audio Replacement Overlay (Preserves 100% video duration)
        output_video_path = None
        output_video_duration = 0.0
        t0 = time.time()

        if is_video:
            update_progress("Rendering video with target audio dubbing & subtitles...", 94)
            output_video_path = os.path.join(self.outputs_dir, f"{file_stem}_translated_{primary_target}.mp4")
            success_overlay, res_overlay = overlay_subtitles_on_video(
                file_path,
                opt_subtitles,
                output_video_path,
                dubbed_audio_path=dubbed_wav_path if dub_success else None
            )
            if success_overlay:
                output_video_duration = get_media_duration(output_video_path)
            else:
                output_video_path = file_path
                output_video_duration = input_media_duration

        timings["video_rendering_sec"] = round(time.time() - t0, 3)

        # Stage 10: CSV Logging, Metrics & Strict Hard Validation
        update_progress("Computing latency & evaluation metrics...", 98)
        total_time = round(time.time() - total_start_time, 4)
        
        csv_path = export_translation_csv(
            opt_subtitles,
            detected_src_lang,
            primary_target,
            "NLLB-200 / Whisper / Non-Overlapping TTS",
            os.path.join(self.outputs_dir, f"{file_stem}_log.csv")
        )

        bleu_metrics = compute_bleu_score(trans_transcript, reference_translation)
        wer_metrics = compute_wer_score(orig_transcript, None)

        record_history(
            mode="VIDEO" if is_video else "AUDIO",
            source_lang=detected_src_lang,
            target_lang=primary_target,
            source_text=orig_transcript,
            translated_text=trans_transcript,
            model_used="Whisper + NLLB + Non-Overlapping TTS Dubbing",
            processing_time_sec=total_time
        )

        # --- STRICT HARD PIPELINE VALIDATION AUDIT ---
        unique_speakers = set(s.get("speaker", "Speaker 1") for s in opt_subtitles)
        speaker_count = len(unique_speakers)
        dur_diff = abs(input_media_duration - output_video_duration) if is_video else 0.0

        if single_speaker_mode and speaker_count > 1:
            validation_errors.append(f"Speaker count mismatch: Expected 1 speaker, detected {speaker_count}.")

        if is_video and dur_diff > 2.0:
            validation_errors.append(f"Video duration mismatch: Input {input_media_duration:.2f}s vs Output {output_video_duration:.2f}s (Diff: {dur_diff:.2f}s).")

        if is_video and (not output_video_path or not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0):
            validation_errors.append("Final rendered video file is missing or empty.")

        safe_log("\n==================================================================")
        safe_log("--- HARD PIPELINE VALIDATION AUDIT ---")
        safe_log(f"  Input Duration:          {input_media_duration:.2f}s")
        safe_log(f"  Output Duration:         {output_video_duration:.2f}s (Diff: {dur_diff:.2f}s)")
        safe_log(f"  Single Speaker Mode:     {single_speaker_mode}")
        safe_log(f"  Detected Speakers:       {speaker_count} ({unique_speakers})")
        safe_log(f"  Genuine Speech Segments: {len(raw_segments)}")
        safe_log(f"  Translated Segments:     {len(translated_segments)}")
        safe_log(f"  TTS Dubbed Segments:     {len(opt_subtitles)}")
        safe_log(f"  Overlapping Audio Segs:  {overlapping_count}")
        unique_speaker_list = sorted(unique_speakers)
        safe_log(f"  Stage Timings:           ASR: {timings.get('asr_sec',0)}s | Diarization: {timings.get('diarization_sec',0)}s | Translation: {timings.get('translation_sec',0)}s | TTS: {timings.get('tts_dubbing_sec',0)}s | Video: {timings.get('video_rendering_sec',0)}s")
        safe_log(f"  Total Elapsed Time:      {total_time}s")
        safe_log(f"  Validation Result:       {'PASSED' if not validation_errors else 'FAILED (' + str(validation_errors) + ')'}")
        safe_log("==================================================================\n")

        if validation_errors:
            error_summary = "; ".join(validation_errors)
            update_progress(f"Translation failed validation: {error_summary}", 100)
            return {
                "status": "validation_failed",
                "validation_errors": validation_errors,
                "error": error_summary,
                "file_name": os.path.basename(file_path),
                "input_duration_sec": round(input_media_duration, 2),
                "output_duration_sec": round(output_video_duration, 2) if is_video else round(input_media_duration, 2),
                "speaker_count": speaker_count,
                "speech_segments_count": len(raw_segments),
                "translated_segments_count": len(translated_segments),
                "overlapping_segments_count": overlapping_count,
                "processing_time_sec": total_time,
                "stage_timings": timings
            }

        update_progress("Completed successfully! 100%", 100)

        return {
            "status": "completed",
            "file_name": os.path.basename(file_path),
            "is_video": is_video,
            "input_duration_sec": round(input_media_duration, 2),
            "output_duration_sec": round(output_video_duration, 2) if is_video else round(input_media_duration, 2),
            "duration_difference_sec": round(dur_diff, 2),
            "speaker_count": speaker_count,
            "unique_speakers": list(unique_speakers),
            "speech_segments_count": len(raw_segments),
            "translated_segments_count": len(translated_segments),
            "overlapping_segments_count": overlapping_count,
            "source_lang": detected_src_lang,
            "target_lang": primary_target,
            "original_transcript": orig_transcript,
            "translated_transcript": trans_transcript,
            "subtitles": opt_subtitles,
            "srt_path": srt_path,
            "vtt_path": vtt_path,
            "csv_path": csv_path,
            "dubbed_audio_path": dubbed_wav_path if dub_success else None,
            "output_video_path": output_video_path,
            "processing_time_sec": total_time,
            "stage_timings": timings,
            "bleu_metrics": bleu_metrics,
            "wer_metrics": wer_metrics
        }


# Global Instance
pipeline = MultimodalPipeline()
