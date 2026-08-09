// --- State Management ---
let pipelineStatus = {
    preprocess_running: false,
    train_running: false,
    has_raw_data: false,
    has_preprocessed: false,
    models: [],
    outputs: []
};

let activePollingInterval = null;
let currentTaskPolling = null; // 'preprocess', 'train', or null

// --- Audio & Visualizer State ---
let audioContext = null;
let analyser = null;
let source = null;
let dataArray = null;
let bufferLength = null;
let animationFrameId = null;
let isAudioConnected = false;

// --- DOM Cache ---
const globalStatusIndicator = document.getElementById('global-status-indicator');
const globalStatusText = document.getElementById('global-status-text');

// Preprocess elements
const btnPreprocess = document.getElementById('btn-preprocess');
const dataStatusBadge = document.getElementById('data-status-badge');

// Training elements
const btnTrain = document.getElementById('btn-train');
const btnStopTrain = document.getElementById('btn-stop-train');
const inputEpochs = document.getElementById('input-epochs');
const inputBatchSize = document.getElementById('input-batch-size');
const modelStatusBadge = document.getElementById('model-status-badge');
const trainProgressContainer = document.getElementById('training-progress-container');
const progressEpoch = document.getElementById('progress-epoch');
const progressLoss = document.getElementById('progress-loss');
const progressAccuracy = document.getElementById('progress-accuracy');
const progressStatusDetail = document.getElementById('progress-status-detail');
const trainProgressFill = document.getElementById('training-progress-fill');

// Generation elements
const btnGenerate = document.getElementById('btn-generate');
const sliderTemp = document.getElementById('slider-temp');
const sliderNotes = document.getElementById('slider-notes');
const valTemp = document.getElementById('val-temp');
const valNotes = document.getElementById('val-notes');
const tempTip = document.getElementById('temp-tip');

// Audio elements
const audioElement = document.getElementById('audio-element');
const playerBtnPlay = document.getElementById('player-btn-play');
const playerTimeline = document.getElementById('player-timeline');
const playerCurrentTime = document.getElementById('player-current-time');
const playerTotalTime = document.getElementById('player-total-time');
const volumeSlider = document.getElementById('volume-slider');
const volumeIcon = document.getElementById('volume-icon');
const linkDownloadMidi = document.getElementById('link-download-midi');
const linkDownloadWav = document.getElementById('link-download-wav');
const visualizerCanvas = document.getElementById('visualizer-canvas');
const visualizerIdleOverlay = document.getElementById('visualizer-idle-overlay');

// Console elements
const consoleOutput = document.getElementById('console-output');
const logActiveTag = document.getElementById('log-active-tag');

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    // Initial UI updates
    updateTempTip(sliderTemp.value);
    
    // Sliders
    sliderTemp.addEventListener('input', (e) => {
        valTemp.textContent = e.target.value;
        updateTempTip(e.target.value);
    });
    sliderNotes.addEventListener('input', (e) => {
        valNotes.textContent = e.target.value;
    });
    
    // Action Buttons
    btnPreprocess.addEventListener('click', startPreprocessing);
    btnTrain.addEventListener('click', startTraining);
    btnStopTrain.addEventListener('click', stopTraining);
    btnGenerate.addEventListener('click', generateMusic);
    
    // Custom Audio Player setup
    setupAudioPlayer();
    
    // Setup visualizer canvas sizes
    setupCanvas();
    window.addEventListener('resize', setupCanvas);
    
    // Start idle visualization
    drawVisualizer();
    
    // Fetch initial status and start general status polling
    fetchStatus();
    setInterval(fetchStatus, 3000);
});

// --- API Interactions ---

function updateTempTip(temp) {
    const val = parseFloat(temp);
    if (val < 0.4) {
        tempTip.textContent = "Strict & Repetitive";
        tempTip.style.borderColor = "rgba(0,229,255,0.3)";
        tempTip.style.color = "var(--color-cyan)";
    } else if (val < 0.8) {
        tempTip.textContent = "Balanced";
        tempTip.style.borderColor = "rgba(255,255,255,0.1)";
        tempTip.style.color = "var(--color-text-muted)";
    } else if (val < 1.2) {
        tempTip.textContent = "Creative & Bold";
        tempTip.style.borderColor = "rgba(255,0,127,0.3)";
        tempTip.style.color = "var(--color-pink)";
    } else {
        tempTip.textContent = "Experimental / Chaotic";
        tempTip.style.borderColor = "rgba(255,159,28,0.3)";
        tempTip.style.color = "var(--color-orange)";
    }
}

async function fetchStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        pipelineStatus = data;
        
        // Update general badges
        if (data.has_preprocessed) {
            dataStatusBadge.textContent = "Preprocessed";
            dataStatusBadge.className = "status-badge complete";
        } else {
            dataStatusBadge.textContent = "Missing";
            dataStatusBadge.className = "status-badge missing";
        }
        
        if (data.models.length > 0) {
            modelStatusBadge.textContent = data.models.includes('best_model.h5') ? "Best Checkpoint Ready" : "Weights Ready";
            modelStatusBadge.className = "status-badge complete";
        } else {
            modelStatusBadge.textContent = "No Models Trained";
            modelStatusBadge.className = "status-badge missing";
        }
        
        // Check if outputs exist to enable player and downloads
        const hasWav = data.outputs.includes('generated_music.wav');
        const hasMidi = data.outputs.includes('generated_music.mid');
        
        if (hasWav) {
            playerBtnPlay.removeAttribute('disabled');
            playerTimeline.removeAttribute('disabled');
            linkDownloadWav.classList.remove('disabled');
            visualizerIdleOverlay.querySelector('p').textContent = "Ready to Play";
        } else {
            playerBtnPlay.setAttribute('disabled', 'true');
            playerTimeline.setAttribute('disabled', 'true');
            linkDownloadWav.classList.add('disabled');
            visualizerIdleOverlay.querySelector('p').textContent = "Generate Music First";
        }
        
        if (hasMidi) {
            linkDownloadMidi.classList.remove('disabled');
        } else {
            linkDownloadMidi.classList.add('disabled');
        }
        
        // Update general state running indicators
        if (data.preprocess_running) {
            setGlobalStatus("running", "Preprocessing classical scores...");
            btnPreprocess.setAttribute('disabled', 'true');
            if (currentTaskPolling !== 'preprocess') startPollingTask('preprocess');
        } else if (data.train_running) {
            setGlobalStatus("running", "Training deep learning model...");
            btnTrain.setAttribute('disabled', 'true');
            btnStopTrain.classList.remove('hidden');
            trainProgressContainer.classList.remove('hidden');
            if (currentTaskPolling !== 'train') startPollingTask('train');
        } else {
            // Idle
            setGlobalStatus("ready", "System Idle / Ready");
            btnPreprocess.removeAttribute('disabled');
            btnTrain.removeAttribute('disabled');
            btnStopTrain.classList.add('hidden');
            
            // Clean up polling if nothing runs
            if (currentTaskPolling !== null && !data.preprocess_running && !data.train_running) {
                stopPollingTask();
            }
        }
        
    } catch (e) {
        console.error("Error fetching status:", e);
        setGlobalStatus("idle", "Server Disconnected");
    }
}

function setGlobalStatus(state, text) {
    globalStatusIndicator.className = "status-indicator " + state;
    globalStatusText.textContent = text;
}

// --- PREPROCESSING ---

async function startPreprocessing() {
    try {
        logActiveTag.textContent = "Preprocessing";
        consoleOutput.textContent = "Starting preprocessing pipeline...\nConnecting to midi download stream...";
        
        const response = await fetch('/api/preprocess', { method: 'POST' });
        const data = await response.json();
        
        if (data.status === 'success') {
            startPollingTask('preprocess');
        } else {
            consoleOutput.textContent += `\nError: ${data.message}`;
        }
    } catch (e) {
        consoleOutput.textContent += `\nConnection Error: ${e.message}`;
    }
}

// --- TRAINING ---

async function startTraining() {
    const epochs = parseInt(inputEpochs.value) || 20;
    const batchSize = parseInt(inputBatchSize.value) || 64;
    
    try {
        logActiveTag.textContent = "Training";
        consoleOutput.textContent = "Compiling neural model architecture...\nInitializing training pipeline...\n";
        trainProgressContainer.classList.remove('hidden');
        trainProgressFill.style.width = '0%';
        
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ epochs, batch_size: batchSize })
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            startPollingTask('train');
        } else {
            consoleOutput.textContent += `\nError: ${data.message}`;
        }
    } catch (e) {
        consoleOutput.textContent += `\nConnection Error: ${e.message}`;
    }
}

async function stopTraining() {
    try {
        consoleOutput.textContent += "\nRequesting manual training abort...\n";
        const response = await fetch('/api/train/stop', { method: 'POST' });
        const data = await response.json();
        consoleOutput.textContent += `\n${data.message}\n`;
        stopPollingTask();
        fetchStatus();
    } catch (e) {
        consoleOutput.textContent += `\nError stopping training: ${e.message}`;
    }
}

// --- POLL DETAILS FOR SUBPROCESSES ---

function startPollingTask(task) {
    if (activePollingInterval) clearInterval(activePollingInterval);
    currentTaskPolling = task;
    
    if (task === 'preprocess') {
        logActiveTag.textContent = "Preprocessing";
        activePollingInterval = setInterval(pollPreprocess, 1000);
    } else if (task === 'train') {
        logActiveTag.textContent = "Training";
        activePollingInterval = setInterval(pollTrain, 1500);
    }
}

function stopPollingTask() {
    if (activePollingInterval) {
        clearInterval(activePollingInterval);
        activePollingInterval = null;
    }
    currentTaskPolling = null;
    logActiveTag.textContent = "Idle";
}

async function pollPreprocess() {
    try {
        const response = await fetch('/api/preprocess/status');
        const data = await response.json();
        
        consoleOutput.textContent = data.logs || "Preprocessing logs waiting...";
        scrollLogsToBottom();
        
        if (data.status === 'completed') {
            consoleOutput.textContent += "\n\n>>> DATA PREPROCESSING COMPLETED SUCCESSFULLY! <<<";
            stopPollingTask();
            fetchStatus();
        } else if (data.status === 'failed') {
            consoleOutput.textContent += "\n\n>>> ERROR: PREPROCESSING FAILED. Check logs above. <<<";
            stopPollingTask();
            fetchStatus();
        }
    } catch (e) {
        console.error("Error polling preprocess:", e);
    }
}

async function pollTrain() {
    try {
        const response = await fetch('/api/train/status');
        const data = await response.json();
        
        consoleOutput.textContent = data.logs || "Training starting...";
        scrollLogsToBottom();
        
        // Progress display
        if (data.progress) {
            const prog = data.progress;
            if (prog.status === 'training' || prog.status === 'completed') {
                const currentEpoch = prog.epoch;
                const totalEpochs = prog.total_epochs;
                const progressPct = totalEpochs > 0 ? (currentEpoch / totalEpochs) * 100 : 0;
                
                progressEpoch.textContent = `Epoch ${currentEpoch} / ${totalEpochs}`;
                progressLoss.textContent = prog.loss ? `Loss: ${prog.loss.toFixed(4)}` : "Loss: --";
                progressAccuracy.textContent = prog.accuracy ? `Acc: ${(prog.accuracy * 100).toFixed(1)}%` : "Accuracy: --";
                trainProgressFill.style.width = `${progressPct}%`;
                progressStatusDetail.textContent = `Running (Epoch ${currentEpoch})`;
            } else if (prog.status === 'stopped') {
                progressStatusDetail.textContent = "Training Stopped";
            }
        }
        
        if (data.status === 'completed') {
            consoleOutput.textContent += "\n\n>>> MODEL TRAINING COMPLETED SUCCESSFULLY! Checkpoints saved. <<<";
            progressStatusDetail.textContent = "Completed Successfully";
            stopPollingTask();
            fetchStatus();
        } else if (data.status === 'failed') {
            consoleOutput.textContent += `\n\n>>> ERROR: TRAINING FAILED. <<<`;
            progressStatusDetail.textContent = "Failed";
            stopPollingTask();
            fetchStatus();
        }
    } catch (e) {
        console.error("Error polling training status:", e);
    }
}

function scrollLogsToBottom() {
    const parent = consoleOutput.parentElement;
    parent.scrollTop = parent.scrollHeight;
}

// --- GENERATION ---

async function generateMusic() {
    const notes = parseInt(sliderNotes.value) || 100;
    const temp = parseFloat(sliderTemp.value) || 0.7;
    const toneSelect = document.getElementById('select-tone-style');
    const tone = toneSelect ? toneSelect.value : 'piano';
    
    try {
        logActiveTag.textContent = "Generation";
        consoleOutput.textContent = `Generating new classical performance...\nNotes: ${notes}\nTemperature: ${temp}\nTone Timbre: ${tone.toUpperCase()}\n\nRunning generator models...`;
        btnGenerate.setAttribute('disabled', 'true');
        
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ notes, temp, tone })
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            consoleOutput.textContent = data.stdout + "\n\n>>> COMPOSITION GENERATED SUCCESSFULLY! Click play below to listen. <<<";
            // Force reload audio file
            audioElement.src = "/api/play?t=" + new Date().getTime();
            audioElement.load();
            
            // Enable visualizer idle mode
            visualizerIdleOverlay.classList.remove('hidden');
            visualizerIdleOverlay.querySelector('p').textContent = "Ready to Play";
            
            fetchStatus();
        } else {
            consoleOutput.textContent = (data.stdout || '') + "\n" + (data.stderr || '') + `\n\nError: ${data.message}`;
        }
    } catch (e) {
        consoleOutput.textContent += `\nConnection Error: ${e.message}`;
    } finally {
        btnGenerate.removeAttribute('disabled');
        scrollLogsToBottom();
    }
}

// --- CUSTOM AUDIO PLAYER ---

function setupAudioPlayer() {
    // Play/Pause Toggle
    playerBtnPlay.addEventListener('click', toggleAudio);
    
    // Audio events
    audioElement.addEventListener('timeupdate', () => {
        if (audioElement.duration) {
            const pct = (audioElement.currentTime / audioElement.duration) * 100;
            playerTimeline.value = pct;
            playerCurrentTime.textContent = formatTime(audioElement.currentTime);
        }
    });
    
    audioElement.addEventListener('loadedmetadata', () => {
        playerTotalTime.textContent = formatTime(audioElement.duration);
    });
    
    audioElement.addEventListener('ended', () => {
        playerBtnPlay.innerHTML = '<i class="fa-solid fa-play"></i>';
        visualizerIdleOverlay.classList.remove('hidden');
    });
    
    // Timeline Seek
    playerTimeline.addEventListener('input', (e) => {
        if (audioElement.duration) {
            const targetTime = (e.target.value / 100) * audioElement.duration;
            audioElement.currentTime = targetTime;
        }
    });
    
    // Volume Control
    volumeSlider.addEventListener('input', (e) => {
        const vol = parseFloat(e.target.value);
        audioElement.volume = vol;
        updateVolumeIcon(vol);
    });
}

function formatTime(secs) {
    const mins = Math.floor(secs / 60);
    const remainingSecs = Math.floor(secs % 60);
    return `${mins}:${remainingSecs < 10 ? '0' : ''}${remainingSecs}`;
}

function updateVolumeIcon(vol) {
    if (vol === 0) {
        volumeIcon.className = "fa-solid fa-volume-xmark text-dim";
    } else if (vol < 0.4) {
        volumeIcon.className = "fa-solid fa-volume-low text-dim";
    } else {
        volumeIcon.className = "fa-solid fa-volume-high text-dim";
    }
}

function toggleAudio() {
    if (audioElement.paused) {
        // Initialize AudioContext on first play interaction
        if (!audioContext) {
            initAudioContext();
        }
        
        audioElement.play();
        playerBtnPlay.innerHTML = '<i class="fa-solid fa-pause"></i>';
        visualizerIdleOverlay.classList.add('hidden');
    } else {
        audioElement.pause();
        playerBtnPlay.innerHTML = '<i class="fa-solid fa-play"></i>';
        visualizerIdleOverlay.classList.remove('hidden');
        visualizerIdleOverlay.querySelector('p').textContent = "Paused";
    }
}

// --- VISUALIZATION CANVAS ---

function setupCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = visualizerCanvas.parentElement.getBoundingClientRect();
    
    visualizerCanvas.width = rect.width * dpr;
    visualizerCanvas.height = rect.height * dpr;
    
    const ctx = visualizerCanvas.getContext('2d');
    ctx.scale(dpr, dpr);
}

function initAudioContext() {
    try {
        window.AudioContext = window.AudioContext || window.webkitAudioContext;
        audioContext = new AudioContext();
        
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 128; // Low resolution for simple dashboard visualizer bars
        bufferLength = analyser.frequencyBinCount;
        dataArray = new Uint8Array(bufferLength);
        
        // Connect HTML Audio source to Web Audio node pipeline
        source = audioContext.createMediaElementSource(audioElement);
        source.connect(analyser);
        analyser.connect(audioContext.destination);
        isAudioConnected = true;
    } catch (e) {
        console.error("Web Audio API not supported / failed to bind:", e);
    }
}

// Draw Loop
function drawVisualizer() {
    animationFrameId = requestAnimationFrame(drawVisualizer);
    
    const canvas = visualizerCanvas;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    
    ctx.clearRect(0, 0, w, h);
    
    // 1. Idle mode or disconnected audio
    if (!isAudioConnected || !analyser || audioElement.paused) {
        // Draw static/sine idle lines
        ctx.strokeStyle = "rgba(157, 78, 221, 0.25)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        const sliceWidth = w / 100;
        let x = 0;
        const time = Date.now() * 0.003;
        
        for (let i = 0; i < 100; i++) {
            const y = (h / 2) + Math.sin(i * 0.15 + time) * 12 * Math.sin(i * 0.03);
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
            x += sliceWidth;
        }
        ctx.stroke();
        return;
    }
    
    // 2. Playback / Active Visualization Mode
    analyser.getByteFrequencyData(dataArray);
    
    const barWidth = (w / bufferLength) * 1.2;
    let barHeight;
    let x = 0;
    
    for (let i = 0; i < bufferLength; i++) {
        barHeight = (dataArray[i] / 255) * h * 0.85;
        
        if (barHeight < 2) barHeight = 2; // Always draw a baseline
        
        // Neon color gradient
        const grad = ctx.createLinearGradient(0, h, 0, h - barHeight);
        grad.addColorStop(0, 'rgba(157, 78, 221, 0.8)'); // Purple base
        grad.addColorStop(1, 'rgba(0, 229, 255, 0.9)');  // Cyan top
        
        ctx.fillStyle = grad;
        
        // Rounded bars
        const rx = x;
        const ry = h - barHeight;
        const rw = barWidth - 3;
        const rh = barHeight;
        
        // Draw path with slight rounded corners on top
        ctx.beginPath();
        ctx.roundRect ? ctx.roundRect(rx, ry, rw, rh, [4, 4, 0, 0]) : ctx.rect(rx, ry, rw, rh);
        ctx.fill();
        
        x += barWidth;
    }
}
