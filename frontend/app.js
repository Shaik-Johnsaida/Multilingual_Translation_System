// Frontend Logic for AI Multilingual Audio Dubbing & Subtitle Generator

const API_BASE = 'http://127.0.0.1:8000/api';

let supportedLanguages = [];
let activeJobId = null;
let pollTimer = null;

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    fetchSystemStatus();
    fetchLanguages();
    initTextMode();
    initAudioMode();
    initVideoMode();
    initHistoryMode();
});

// Tab Switcher
function initTabs() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            navButtons.forEach(b => b.classList.remove('active'));
            tabPanels.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const tabId = btn.getAttribute('data-tab');
            const panel = document.getElementById(tabId);
            if (panel) panel.classList.add('active');

            if (tabId === 'history-mode') fetchHistory();
            if (tabId === 'languages-mode') renderLanguagesMatrix();
        });
    });
}

// Fetch Hardware System Status
async function fetchSystemStatus() {
    try {
        const res = await fetch(`${API_BASE}/status`);
        const data = await res.json();
        if (data.device) {
            document.getElementById('badge-text').textContent = `Local AI Engine Ready (${data.device.toUpperCase()})`;
        }
    } catch (e) {
        console.warn('Backend server connecting...', e);
    }
}

// Fetch Centralized Dynamic Language Registry from Backend (EXACTLY 10 LANGUAGES)
async function fetchLanguages() {
    try {
        const res = await fetch(`${API_BASE}/languages`);
        const data = await res.json();
        supportedLanguages = data.languages || [];
        populateLanguageSelects();
    } catch (e) {
        console.warn('Using fallback language registry:', e);
        supportedLanguages = [
            { flores_code: 'tel_Telu', name: 'Telugu', code: 'te', script: 'Telugu' },
            { flores_code: 'hin_Deva', name: 'Hindi', code: 'hi', script: 'Devanagari' },
            { flores_code: 'eng_Latn', name: 'English', code: 'en', script: 'Latin' },
            { flores_code: 'urd_Arab', name: 'Urdu', code: 'ur', script: 'Arabic' },
            { flores_code: 'tam_Taml', name: 'Tamil', code: 'ta', script: 'Tamil' },
            { flores_code: 'mal_Mlym', name: 'Malayalam', code: 'ml', script: 'Malayalam' },
            { flores_code: 'zho_Hans', name: 'Chinese (Mandarin)', code: 'zh', script: 'Simplified Han' },
            { flores_code: 'jpn_Jpan', name: 'Japanese', code: 'ja', script: 'Japanese' },
            { flores_code: 'spa_Latn', name: 'Spanish', code: 'es', script: 'Latin' },
            { flores_code: 'fra_Latn', name: 'French', code: 'fr', script: 'Latin' }
        ];
        populateLanguageSelects();
    }
}

function populateLanguageSelects() {
    const selects = [
        { id: 'text-src-lang', includeAuto: true, defaultVal: 'auto' },
        { id: 'text-tgt-lang', includeAuto: false, defaultVal: 'tel_Telu' },
        { id: 'audio-src-lang', includeAuto: true, defaultVal: 'auto' },
        { id: 'audio-tgt-lang', includeAuto: false, defaultVal: 'tel_Telu' },
        { id: 'video-src-lang', includeAuto: true, defaultVal: 'auto' },
        { id: 'video-tgt-lang', includeAuto: false, defaultVal: 'tel_Telu' }
    ];

    selects.forEach(({ id, includeAuto, defaultVal }) => {
        const elem = document.getElementById(id);
        if (!elem) return;
        elem.innerHTML = '';

        if (includeAuto) {
            const optAuto = document.createElement('option');
            optAuto.value = 'auto';
            optAuto.textContent = '🔍 Auto Detect';
            elem.appendChild(optAuto);
        }

        supportedLanguages.forEach(lang => {
            const opt = document.createElement('option');
            opt.value = lang.flores_code;
            opt.textContent = `${lang.name} (${lang.code})`;
            if (lang.flores_code === defaultVal) opt.selected = true;
            elem.appendChild(opt);
        });
    });

    renderLanguagesMatrix();
}

// -------------------------------------------------------------
// TEXT TRANSLATION MODE
// -------------------------------------------------------------
function initTextMode() {
    const txtInput = document.getElementById('text-input');
    const txtOutput = document.getElementById('text-output-display');
    const charCount = document.getElementById('char-count');
    const btnTranslate = document.getElementById('btn-translate-text');
    const btnClear = document.getElementById('btn-clear-text');
    const btnCopy = document.getElementById('btn-copy-text');
    const btnSwap = document.getElementById('btn-swap-lang');

    txtInput.addEventListener('input', () => {
        charCount.textContent = `${txtInput.value.length} / 5000 characters`;
    });

    btnClear.addEventListener('click', () => {
        txtInput.value = '';
        txtOutput.innerHTML = '<span class="placeholder-text">Translation output will appear here...</span>';
        charCount.textContent = '0 / 5000 characters';
    });

    btnCopy.addEventListener('click', () => {
        const text = txtOutput.innerText;
        if (text && text !== 'Translation output will appear here...') {
            navigator.clipboard.writeText(text);
            btnCopy.textContent = 'Copied!';
            setTimeout(() => btnCopy.textContent = 'Copy', 2000);
        }
    });

    btnSwap.addEventListener('click', () => {
        const srcSelect = document.getElementById('text-src-lang');
        const tgtSelect = document.getElementById('text-tgt-lang');

        if (srcSelect.value !== 'auto') {
            const temp = srcSelect.value;
            srcSelect.value = tgtSelect.value;
            tgtSelect.value = temp;
        }
    });

    btnTranslate.addEventListener('click', async () => {
        const text = txtInput.value.trim();
        if (!text) return;

        btnTranslate.disabled = true;
        btnTranslate.textContent = 'Translating...';
        txtOutput.innerHTML = '<span class="placeholder-text">Processing local neural translation...</span>';

        try {
            const res = await fetch(`${API_BASE}/translate/text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    source_lang: document.getElementById('text-src-lang').value,
                    target_lang: document.getElementById('text-tgt-lang').value
                })
            });

            const data = await res.json();
            if (data.status === 'success' && data.result) {
                txtOutput.textContent = data.result.translated_text;
                document.getElementById('model-meta-info').textContent = `Model: ${data.result.model_used} (${data.result.processing_time_sec}s)`;
            } else {
                txtOutput.textContent = 'Translation failed.';
            }
        } catch (e) {
            txtOutput.textContent = `Error: Backend server offline. (${e.message})`;
        } finally {
            btnTranslate.disabled = false;
            btnTranslate.textContent = 'Translate Now';
        }
    });
}

// -------------------------------------------------------------
// AUDIO TRANSLATION & DUBBING MODE
// -------------------------------------------------------------
function initAudioMode() {
    const fileInput = document.getElementById('audio-file-input');
    const fileNameBadge = document.getElementById('audio-file-name');
    const btnProcess = document.getElementById('btn-process-audio');
    const origPlayerContainer = document.getElementById('audio-orig-player-container');
    const origPlayer = document.getElementById('audio-orig-player');

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            fileNameBadge.textContent = `Selected: ${file.name}`;
            origPlayer.src = URL.createObjectURL(file);
            origPlayerContainer.classList.remove('hidden');
        }
    });

    btnProcess.addEventListener('click', async () => {
        if (!fileInput.files.length) {
            alert('Please select an audio file first.');
            return;
        }
        uploadAndProcessMedia(fileInput.files[0], 'audio');
    });
}

// -------------------------------------------------------------
// VIDEO SUBTITLE & AUDIO DUBBING MODE
// -------------------------------------------------------------
function initVideoMode() {
    const fileInput = document.getElementById('video-file-input');
    const fileNameBadge = document.getElementById('video-file-name');
    const btnProcess = document.getElementById('btn-process-video');
    const origPlayerContainer = document.getElementById('video-orig-player-container');
    const origPlayer = document.getElementById('video-orig-player');

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            fileNameBadge.textContent = `Selected: ${file.name}`;
            origPlayer.src = URL.createObjectURL(file);
            origPlayerContainer.classList.remove('hidden');
        }
    });

    btnProcess.addEventListener('click', async () => {
        if (!fileInput.files.length) {
            alert('Please select a video file first.');
            return;
        }
        uploadAndProcessMedia(fileInput.files[0], 'video');
    });
}

// Media Upload & Async Job Tracking
async function uploadAndProcessMedia(file, type) {
    const srcLang = document.getElementById(`${type}-src-lang`).value;
    const tgtLang = document.getElementById(`${type}-tgt-lang`).value;

    const progressContainer = document.getElementById(`${type}-progress-container`);
    const progressFill = document.getElementById(`${type}-progress-fill`);
    const stageText = document.getElementById(`${type}-stage-text`);
    const pctText = document.getElementById(`${type}-pct-text`);
    const resultsContainer = document.getElementById(`${type}-results`);

    progressContainer.classList.remove('hidden');
    resultsContainer.classList.add('hidden');
    progressFill.style.width = '5%';
    stageText.textContent = 'Uploading media...';
    pctText.textContent = '5%';

    const formData = new FormData();
    formData.append('file', file);
    formData.append('source_lang', srcLang);
    formData.append('target_lang', tgtLang);

    try {
        const res = await fetch(`${API_BASE}/process/media`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (data.job_id) {
            activeJobId = data.job_id;
            pollJobStatus(activeJobId, type);
        }
    } catch (e) {
        stageText.textContent = `Upload failed: ${e.message}`;
    }
}

function pollJobStatus(jobId, type) {
    if (pollTimer) clearInterval(pollTimer);

    pollTimer = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/jobs/${jobId}`);
            const job = await res.json();

            const progressFill = document.getElementById(`${type}-progress-fill`);
            const stageText = document.getElementById(`${type}-stage-text`);
            const pctText = document.getElementById(`${type}-pct-text`);

            if (job.progress_pct) {
                progressFill.style.width = `${job.progress_pct}%`;
                pctText.textContent = `${job.progress_pct}%`;
                stageText.textContent = job.stage || 'Processing...';
            }

            if (job.status === 'completed') {
                clearInterval(pollTimer);
                displayResults(job.result, type);
            } else if (job.status === 'failed') {
                clearInterval(pollTimer);
                stageText.textContent = `Processing Error: ${job.error || 'Failed'}`;
            }
        } catch (e) {
            console.error('Job polling error:', e);
        }
    }, 1000);
}

function displayResults(result, type) {
    const resultsContainer = document.getElementById(`${type}-results`);
    resultsContainer.classList.remove('hidden');

    if (type === 'audio') {
        document.getElementById('audio-orig-transcript').textContent = result.original_transcript;
        document.getElementById('audio-trans-transcript').textContent = result.translated_transcript;
        
        const dubbedPlayer = document.getElementById('audio-dubbed-player');
        if (result.dubbed_audio_path && dubbedPlayer) {
            const audioFileName = result.dubbed_audio_path.split('\\').pop().split('/').pop();
            dubbedPlayer.src = `${API_BASE}/downloads/${audioFileName}`;
        }
    } else if (type === 'video') {
        const videoPlayer = document.getElementById('video-player');
        const videoSource = document.getElementById('video-source');
        
        if (result.output_video_path) {
            const videoFileName = result.output_video_path.split('\\').pop().split('/').pop();
            videoSource.src = `${API_BASE}/downloads/${videoFileName}`;
            videoPlayer.load();
        }

        const subList = document.getElementById('subtitle-list');
        subList.innerHTML = '';

        (result.subtitles || []).forEach(sub => {
            const item = document.createElement('div');
            item.className = 'sub-item';
            item.innerHTML = `
                <div class="sub-time">${sub.start_srt} --> ${sub.end_srt}</div>
                <div class="sub-speaker">${sub.speaker}</div>
                <div>${sub.translated_text}</div>
            `;
            subList.appendChild(item);
        });

        // Downloads
        document.getElementById('btn-dl-srt').onclick = () => window.open(`${API_BASE}/downloads/${result.srt_path.split('\\').pop()}`);
        document.getElementById('btn-dl-vtt').onclick = () => window.open(`${API_BASE}/downloads/${result.vtt_path.split('\\').pop()}`);
        document.getElementById('btn-dl-csv').onclick = () => window.open(`${API_BASE}/downloads/${result.csv_path.split('\\').pop()}`);
        if (result.output_video_path) {
            document.getElementById('btn-dl-video').onclick = () => window.open(`${API_BASE}/downloads/${result.output_video_path.split('\\').pop()}`);
        }
    }
}

// -------------------------------------------------------------
// SUPPORTED LANGUAGES MATRIX (10 LANGUAGES)
// -------------------------------------------------------------
function renderLanguagesMatrix() {
    const tbody = document.getElementById('languages-table-body');
    const searchInput = document.getElementById('search-lang-matrix');
    if (!tbody) return;

    const filterText = (searchInput?.value || '').toLowerCase();
    tbody.innerHTML = '';

    supportedLanguages.forEach(lang => {
        if (filterText && !lang.name.toLowerCase().includes(filterText) && !lang.flores_code.toLowerCase().includes(filterText)) {
            return;
        }

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${lang.name}</strong></td>
            <td><code>${lang.code}</code> (${lang.flores_code})</td>
            <td>${lang.script || 'Latin'}</td>
            <td><span class="status-dot green"></span> Supported</td>
            <td><span class="status-dot green"></span> Supported</td>
            <td><span class="status-dot green"></span> Supported</td>
            <td><span class="status-dot green"></span> Supported</td>
        `;
        tbody.appendChild(tr);
    });

    if (searchInput) {
        searchInput.oninput = () => renderLanguagesMatrix();
    }
}

// -------------------------------------------------------------
// TRANSLATION HISTORY
// -------------------------------------------------------------
function initHistoryMode() {
    const btnRef = document.getElementById('btn-refresh-history');
    if (btnRef) btnRef.addEventListener('click', fetchHistory);
}

async function fetchHistory() {
    const tbody = document.getElementById('history-table-body');
    if (!tbody) return;

    tbody.innerHTML = '<tr><td colspan="7">Loading history...</td></tr>';
    try {
        const res = await fetch(`${API_BASE}/history`);
        const data = await res.json();
        tbody.innerHTML = '';

        (data.history || []).forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.timestamp}</td>
                <td><code>${row.mode}</code></td>
                <td>${row.source_lang} -> ${row.target_lang}</td>
                <td>${row.source_text}</td>
                <td>${row.translated_text}</td>
                <td><small>${row.model_used}</small></td>
                <td>${row.processing_time_sec}s</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        tbody.innerHTML = '<tr><td colspan="7">Could not connect to history storage.</td></tr>';
    }
}
