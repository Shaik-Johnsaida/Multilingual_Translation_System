# 🎬 Real-Time Multilingual Translation & Audio Dubbing System

An **AI-Based Multimodal Audio-Visual Translation, Subtitle & Audio Dubbing Engine** built with Python and PyTorch. It runs **100% locally on your machine with zero external API keys, zero cloud dependencies, and zero costs**. 

It handles text translation, speech transcription, speaker identification, translation, and generates a dubbed voice track perfectly synchronized with the original video!

---

## 🌟 Key Features & Capabilities

### 1. 💬 Universal Text Translation
*   **10 Production-Ready Languages**: 
    *   **Indian Languages**: Hindi (`hi`), Telugu (`te`), Tamil (`ta`), Malayalam (`ml`), Urdu (`ur`)
    *   **Global Languages**: English (`en`), Chinese/Mandarin (`zh`), Japanese (`ja`), Spanish (`es`), French (`fr`)
*   **Auto-Language Detection**: Simply paste text and let the AI automatically identify the source language.
*   **Multi-Target Translation**: Translate a single source text into multiple target languages simultaneously.

### 2. 🎙️ Audio Dubbing & Translation (Audio Mode)
*   **Smart Speech Detection (VAD)**: Automatically detects when someone is speaking and skips silence or background noise.
*   **Whisper Speech-to-Text (STT)**: Transcribes speech locally using OpenAI's Whisper model.
*   **Who-Spoke-When Detection (Speaker Diarization)**:
    *   **Automatic Speaker Clustering**: Groups audio segments by individual voice characteristics so each speaker gets a consistent translated voice.
    *   **Forced Single-Speaker Mode**: Ensures all spoken segments are labeled and voiced as a single speaker.
*   **Interactive Player**: Play original and translated audio files side-by-side.

### 3. 🎬 Video Dubbing & Subtitle Overlay (Video Mode)
*   **Pristine Subtitle Generation**: Outputs synchronized SubRip (`.srt`) and WebVTT (`.vtt`) files.
*   **Non-Overlapping Audio Dubbing**: Generates natural-sounding voice tracks in the target language.
    *   **Per-Speaker Consistent Voice**: Keeps voices consistent per speaker ID across the whole file.
    *   **Intelligent Tempo Stretching**: Speeds up or slows down speech slightly (clamped between 0.85x and 1.45x using FFmpeg `atempo`) so it fits perfectly into the original time slot.
    *   **Zero-Overlap Guarantee**: Adjusts speech start times to prevent speakers from talking over one another.
*   **FFmpeg Burn-in Video Overlay**: Merges the new dubbed audio track and hardcoded subtitles directly onto the video.
*   **100% Video Duration Match**: Preserves the exact length of the original master video.

### 4. 📈 Evaluation Metrics & Storage
*   **Automated Quality Score**: Calculates **BLEU** (translation accuracy) and **WER** (Speech-to-Text accuracy) metrics.
*   **Performance Diagnostics**: Tracks processing latency and the **Real-Time Factor (RTF)** to show how fast the AI runs compared to the media length.
*   **SQLite History database**: Keeps a local record of all translation and dubbing operations for later review.

---

## 📁 System Architecture

Here is how your media file travels through the offline AI pipeline:

```text
                  +-----------------------------------+
                  |      Your Media File (AV/Text)    |
                  +-----------------+-----------------+
                                    |
            +-----------------------+-----------------------+
            |                       |                       |
            v                       v                       v
      [ TEXT MODE ]           [ AUDIO MODE ]          [ VIDEO MODE ]
            |                       |                       |
     Auto Detection                 |                       |
            |                       v                       v
            |             Extract Audio Track     Extract Audio Track
            |                       |                       |
            |                       v                       v
            |             Voice Activity (VAD)    Voice Activity (VAD)
            |                       |                       |
            |                       v                       v
            |                 Whisper STT             Whisper STT
            |                       |                       |
            |                       v                       v
            |             Speaker Identification  Speaker Identification
            |                       |                       |
            +-----------------------+-----------------------+
                                    |
                                    v
                           Machine Translation
                        (NLLB-200 Local Model)
                                    |
                  +-----------------+-----------------+
                  |                                   |
                  v                                   v
         [ Text Output ]                    [ Subtitle Segments ]
                                                      |
                                           +----------+----------+
                                           |                     |
                                           v                     v
                                   SRT/WebVTT Files       TTS Speech Dubbing
                                   (.srt / .vtt)           (Per-Speaker Voice)
                                                                 |
                                                                 v
                                                        FFmpeg Video Merge
                                                      (Burn-in Subtitles &
                                                        Dubbed Audio track)
                                                                 |
                                                                 v
                                                      [ Final Dubbed Video ]
                                                          (.mp4 Output)
```

---

## 🛠️ Requirements & Setup

### Prerequisites
Before running the system, make sure you have the following installed on your computer:
1.  **Python 3.8 or higher**
2.  **FFmpeg**: Critical for extracting, editing, and stitching audio and video tracks.
    *   *Windows*: Install via `winget install Gyan.FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org).
    *   *macOS*: Install via Homebrew: `brew install ffmpeg`.
    *   *Linux*: Install via APT: `sudo apt install ffmpeg`.

### Step 1: Install Dependencies
Open your terminal inside the project directory and run:
```bash
pip install -r requirements.txt
```

> [!NOTE]
> The Text-to-Speech engine uses `gTTS` (Google Text-to-Speech) for high-quality audio synthesis, with a fallback to `pyttsx3` for completely offline local speech. If you encounter missing package errors, run:
> ```bash
> pip install gTTS pyttsx3
> ```

### Step 2: Verify the Installation (Optional)
To verify that all the local AI components, registries, and libraries are functioning properly, run the automated test suite:
```bash
python -m pytest tests/test_system.py -v
```

### Step 3: Launch the App
To start the FastAPI backend and open the modern Web Dashboard, run:
```bash
python run_server.py
```
This command will launch the server and automatically open your default web browser to the interactive dashboard:
**`http://127.0.0.1:8000/dashboard/index.html`**

---

## 🖥️ Using the Web Dashboard

The application features a sleek dark-themed Web Dashboard split into five interactive tabs:

1.  **💬 Text Mode**:
    *   Type or paste text in the left panel.
    *   Select your source language (or leave on **Auto Detect**).
    *   Select a target language and click **Translate Now**.
    *   Copy the output using the **Copy** button.
2.  **🎙️ Audio Dubbing & Translation**:
    *   Drag and drop or select an audio file (MP3, WAV, FLAC, etc.).
    *   Select target language and click **Start Audio Dubbing**.
    *   Watch the real-time progress bar move through VAD, STT, and Dubbing.
    *   Play the translated audio using the built-in media player!
3.  **🎬 Video Dubbing & Subtitles**:
    *   Upload any video file (MP4, MKV, AVI, etc.).
    *   Click **Generate Dubbed Video & Subtitles**.
    *   Once processing finishes, watch the dubbed video preview.
    *   Download the synchronized `.srt` or `.vtt` subtitles, the `.csv` metric log, or the final `.mp4` video with burnt-in subtitles and dubbed audio.
4.  **🌐 Supported Languages**:
    *   View details about the 10 production-supported languages.
5.  **📜 History**:
    *   Review past translation jobs, execution times, and utilized AI models.

---

## 📊 Understanding Quality Metrics (For Beginners)

When processing audio or video, the system automatically evaluates the outputs and stores them in a CSV log:
*   **BLEU Score (Bilingual Evaluation Understudy)**: A score from 0 to 100 measuring how close the AI translation is to a reference text. Higher is better!
*   **WER (Word Error Rate)**: Measures transcription mistakes by comparing spoken words to recognized text. Lower is better (0.0 means perfect transcription).
*   **Real-Time Factor (RTF)**: The ratio of processing time to media duration. An RTF of `0.5` means a 10-second video was processed in 5 seconds.
