document.addEventListener('DOMContentLoaded', () => {
    // ---- 1. DOM Element References ----
    const elements = {
        webcam: document.getElementById('webcam'),
        captionText: document.getElementById('caption-text'),
        confidenceFill: document.getElementById('confidence-fill'),
        confidenceText: document.getElementById('confidence-text'),
        captionTimestamp: document.getElementById('caption-timestamp'),
        startButton: document.getElementById('startButton'),
        stopButton: document.getElementById('stopButton'),
        muteButton: document.getElementById('muteButton'),
        settingsButton: document.getElementById('settingsButton'),
        fullscreenButton: document.getElementById('fullscreenButton'),
        connectionStatus: document.getElementById('connection-status'),
        fpsCounter: document.getElementById('fps-counter'),
        recordingIndicator: document.getElementById('recording-indicator'),
        latencyValue: document.getElementById('latency-value'),
        accuracyValue: document.getElementById('accuracy-value'),
        processedFrames: document.getElementById('processed-frames'),
        captionsCount: document.getElementById('captions-count'),
        historyList: document.getElementById('history-list'),
        deviceInfo: document.getElementById('device-info'),
        resolutionInfo: document.getElementById('resolution-info'),
        cacheInfo: document.getElementById('cache-info'),
        settingsModal: document.getElementById('settingsModal'),
        closeSettings: document.getElementById('closeSettings'),
        saveSettings: document.getElementById('saveSettings'),
        resetSettings: document.getElementById('resetSettings'),
        frameRateSelect: document.getElementById('frameRateSelect'),
        qualitySlider: document.getElementById('qualitySlider'),
        qualityValue: document.getElementById('qualityValue'),
        audioToggle: document.getElementById('audioToggle'),
        statusMessage: document.getElementById('status-message'),
        toastContainer: document.getElementById('toastContainer')
    };

    // ---- 2. Application State & Settings ----
    let socket;
    let stream;
    let frameSenderInterval;
    let isCapturing = false;
    let captionHistory = [];
    
    let settings = {
        frameRate: 15,
        quality: 0.7,
        audio: true
    };

    let performance = {
        sentFrames: 0,
        receivedFrames: 0,
        captionsGenerated: 0,
        totalConfidence: 0,
        startTime: 0,
        latencyBuffer: []
    };
    
    const LATENCY_BUFFER_SIZE = 20;

    // ---- 3. Core Application Logic ----

    /**
     * Starts the video analysis process.
     */
    const startAnalysis = async () => {
        if (isCapturing) return;

        try {
            // Get webcam stream
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 30 }
                },
                audio: false
            });
            elements.webcam.srcObject = stream;
            await elements.webcam.play();
            isCapturing = true;

            // Update UI
            updateUIForStartState();
            connectSocket();

        } catch (err) {
            console.error("Error accessing webcam:", err);
            showToast("Webcam Error", "Could not access the webcam. Please check permissions.", "error");
            updateUIForStopState();
        }
    };

    /**
     * Stops the video analysis process.
     */
    const stopAnalysis = () => {
        if (!isCapturing) return;

        // Stop intervals and streams
        clearInterval(frameSenderInterval);
        frameSenderInterval = null;
        stream?.getTracks().forEach(track => track.stop());
        socket?.disconnect();
        
        // Cancel any ongoing speech
        window.speechSynthesis.cancel();

        // Reset state
        isCapturing = false;
        elements.webcam.srcObject = null;
        updateUIForStopState();
        showToast("Analysis Stopped", "Real-time captioning has been turned off.", "info");
    };

    /**
     * Connects to the WebSocket server and sets up event listeners.
     */
    const connectSocket = () => {
        // Use the current host and port, but with the ws:// protocol
        socket = io(window.location.origin, {
            transports: ['websocket'],
            upgrade: false
        });

        socket.on('connect', () => {
            console.log('Connected to server! SID:', socket.id);
            elements.connectionStatus.textContent = "Connected";
            elements.connectionStatus.style.color = 'var(--success-color)';
            showToast("Connected", "Successfully connected to the AI server.", "success");
            startFrameSending();
        });

        socket.on('caption', handleCaption);

        socket.on('disconnect', () => {
            console.log('Disconnected from server.');
            elements.connectionStatus.textContent = "Disconnected";
            elements.connectionStatus.style.color = 'var(--danger-color)';
            if (isCapturing) {
                stopAnalysis();
            }
        });

        socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            showToast("Connection Error", "Failed to connect to the server.", "error");
            stopAnalysis();
        });
    };

    /**
     * Initializes the interval for sending video frames to the server.
     */
    const startFrameSending = () => {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d', { alpha: false });

        frameSenderInterval = setInterval(() => {
            if (!isCapturing || elements.webcam.paused || elements.webcam.ended) {
                return;
            }
            // Match the server's expected image size
            canvas.width = 384;
            canvas.height = 384;

            context.drawImage(elements.webcam, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL('image/jpeg', settings.quality);
            socket.emit('image', dataUrl);
            performance.sentFrames++;
            updatePerformanceUI();

        }, 1000 / settings.frameRate);
    };

    // ---- 4. UI Update Functions ----

    /**
     * Handles incoming captions from the server.
     * @param {object} data - The caption data from the server.
     */
    const handleCaption = (data) => {
        performance.receivedFrames++;
        performance.captionsGenerated++;
        performance.totalConfidence += data.confidence;
        
        // Update main caption display
        elements.captionText.textContent = data.caption;
        const confidencePercent = (data.confidence * 100).toFixed(0);
        elements.confidenceFill.style.width = `${confidencePercent}%`;
        elements.confidenceText.textContent = `${confidencePercent}%`;
        const timestamp = new Date(data.timestamp * 1000);
        elements.captionTimestamp.textContent = timestamp.toLocaleTimeString();
        
        // Calculate latency
        const latency = (Date.now() / 1000) - data.timestamp;
        performance.latencyBuffer.push(latency);
        if (performance.latencyBuffer.length > LATENCY_BUFFER_SIZE) {
            performance.latencyBuffer.shift();
        }

        // Add to history
        updateHistory(data.caption, confidencePercent, timestamp);

        // Speak the caption
        if (settings.audio) {
            speakCaption(data.caption);
        }
    };
    
    /**
     * Updates the UI to reflect the "capturing started" state.
     */
    const updateUIForStartState = () => {
        elements.startButton.disabled = true;
        elements.stopButton.disabled = false;
        elements.recordingIndicator.classList.add('active');
        elements.statusMessage.textContent = "AI analysis is active...";
        
        // Reset performance metrics
        performance = {
            sentFrames: 0,
            receivedFrames: 0,
            captionsGenerated: 0,
            totalConfidence: 0,
            startTime: Date.now(),
            latencyBuffer: []
        };
        elements.resolutionInfo.textContent = `${elements.webcam.videoWidth}x${elements.webcam.videoHeight}`;
        elements.historyList.innerHTML = '<div class="history-item"><div class="history-text">Waiting for captions...</div></div>';
    };

    /**
     * Updates the UI to reflect the "capturing stopped" state.
     */
    const updateUIForStopState = () => {
        elements.startButton.disabled = false;
        elements.stopButton.disabled = true;
        elements.recordingIndicator.classList.remove('active');
        elements.statusMessage.textContent = "Ready to start analysis.";
        elements.connectionStatus.textContent = "Disconnected";
        elements.connectionStatus.style.color = 'var(--text-secondary)';
        elements.fpsCounter.textContent = '0';
        elements.latencyValue.textContent = '0ms';
        elements.captionText.textContent = "Analysis stopped.";
    };
    
    /**
     * Periodically updates performance metrics on the UI.
     */
    const updatePerformanceUI = () => {
        const elapsedSeconds = (Date.now() - performance.startTime) / 1000;
        if (elapsedSeconds === 0) return;
        
        const fps = (performance.sentFrames / elapsedSeconds).toFixed(0);
        elements.fpsCounter.textContent = fps;
        
        const avgLatency = performance.latencyBuffer.reduce((a, b) => a + b, 0) / performance.latencyBuffer.length || 0;
        elements.latencyValue.textContent = `${(avgLatency * 1000).toFixed(0)}ms`;
        
        const avgConfidence = (performance.totalConfidence / performance.captionsGenerated * 100) || 0;
        elements.accuracyValue.textContent = `${avgConfidence.toFixed(0)}%`;
        
        elements.processedFrames.textContent = performance.receivedFrames;
        elements.captionsCount.textContent = performance.captionsGenerated;
    };
    
    /**
     * Adds a new caption to the history panel.
     * @param {string} caption - The caption text.
     * @param {string} confidence - The confidence percentage string.
     * @param {Date} timestamp - The Date object for the caption.
     */
    const updateHistory = (caption, confidence, timestamp) => {
        // Remove placeholder if it exists
        if (captionHistory.length === 0) {
            elements.historyList.innerHTML = '';
        }

        const historyItem = { caption, confidence, timestamp };
        captionHistory.unshift(historyItem);
        if (captionHistory.length > 20) { // Limit history size
            captionHistory.pop();
        }

        const itemElement = document.createElement('div');
        itemElement.className = 'history-item';
        itemElement.innerHTML = `
            <div class="history-text">${caption}</div>
            <div class="history-meta">
                <span class="history-confidence">${confidence}%</span>
                <span class="history-time">${timestamp.toLocaleTimeString()}</span>
            </div>
        `;
        elements.historyList.prepend(itemElement);

        // Remove the last element if list is too long
        if (elements.historyList.children.length > 20) {
            elements.historyList.lastChild.remove();
        }
    };

    /**
     * Fetches and displays system status from the server.
     */
    const fetchStatus = async () => {
        try {
            const response = await fetch('/status');
            const data = await response.json();
            elements.deviceInfo.textContent = data.device.toUpperCase();
            elements.cacheInfo.textContent = `${(data.performance.cache_hit_rate * 100).toFixed(0)}%`;
        } catch (error) {
            console.error("Error fetching server status:", error);
            elements.deviceInfo.textContent = 'Error';
        }
    };
    
    // ---- 5. Feature Logic (Audio, Settings, etc.) ----

    /**
     * Uses the Web Speech API to speak the provided text.
     * @param {string} text - The text to be spoken.
     */
    const speakCaption = (text) => {
        if (!text || text.toLowerCase().includes("processing")) return;

        window.speechSynthesis.cancel(); // Interrupt previous speech for the latest update
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.1;
        utterance.pitch = 1.0;
        utterance.volume = 0.8;
        window.speechSynthesis.speak(utterance);
    };

    /**
     * Toggles the settings modal visibility.
     */
    const toggleSettingsModal = () => {
        elements.settingsModal.classList.toggle('active');
    };

    /**
     * Saves the settings from the modal and applies them.
     */
    const saveSettings = () => {
        settings.frameRate = parseInt(elements.frameRateSelect.value, 10);
        settings.quality = parseFloat(elements.qualitySlider.value);
        settings.audio = elements.audioToggle.checked;
        
        toggleSettingsModal();
        showToast("Settings Saved", "Your new settings have been applied.", "success");

        // If capturing, restart the interval to apply new frame rate
        if (isCapturing) {
            clearInterval(frameSenderInterval);
            startFrameSending();
        }
    };

    /**
     * Creates and displays a toast notification.
     * @param {string} title - The title of the toast.
     * @param {string} message - The message body of the toast.
     * @param {string} type - The type of toast (success, error, info, warning).
     */
    const showToast = (title, message, type = 'info') => {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close">&times;</button>
        `;
        elements.toastContainer.appendChild(toast);
        
        setTimeout(() => toast.classList.add('show'), 10);
        
        const removeToast = () => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 500);
        };
        
        toast.querySelector('.toast-close').onclick = removeToast;
        setTimeout(removeToast, 5000);
    };


    // ---- 6. Event Listeners ----
    elements.startButton.addEventListener('click', startAnalysis);
    elements.stopButton.addEventListener('click', stopAnalysis);
    
    elements.settingsButton.addEventListener('click', toggleSettingsModal);
    elements.closeSettings.addEventListener('click', toggleSettingsModal);
    elements.saveSettings.addEventListener('click', saveSettings);

    elements.muteButton.addEventListener('click', () => {
        settings.audio = !settings.audio;
        elements.audioToggle.checked = settings.audio;
        elements.muteButton.classList.toggle('active', settings.audio);
        showToast("Audio " + (settings.audio ? "Enabled" : "Disabled"), "", "info");
    });
    
    elements.fullscreenButton.addEventListener('click', () => {
        if (!document.fullscreenElement) {
            elements.webcam.parentElement.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    });

    elements.qualitySlider.addEventListener('input', (e) => {
        elements.qualityValue.textContent = `${Math.round(e.target.value * 100)}%`;
    });

    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

        switch (e.code) {
            case 'Space':
                e.preventDefault();
                isCapturing ? stopAnalysis() : startAnalysis();
                break;
            case 'KeyS':
                e.preventDefault();
                toggleSettingsModal();
                break;
            case 'KeyM':
                e.preventDefault();
                elements.muteButton.click();
                break;
            case 'KeyF':
                 e.preventDefault();
                elements.fullscreenButton.click();
                break;
        }
    });

    // ---- 7. Initialization ----
    const init = () => {
        updateUIForStopState();
        fetchStatus();
    };

    init();
});