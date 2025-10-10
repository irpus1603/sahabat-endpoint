class DetectionWebSocket {
    constructor(cameraId, containerId, options = {}) {
        this.cameraId = cameraId;
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.socket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = options.maxReconnectAttempts || 5;
        this.reconnectDelay = options.reconnectDelay || 3000;
        this.detectionType = options.detectionType || 'security_intrusion';
        
        // UI elements
        this.videoElement = null;
        this.statusElement = null;
        this.controlsElement = null;
        
        this.init();
    }

    init() {
        this.createUI();
        this.connect();
    }

    createUI() {
        if (!this.container) {
            console.error(`Container with ID ${this.containerId} not found`);
            return;
        }

        this.container.innerHTML = `
            <div class="detection-stream-container">
                <div class="video-container">
                    <canvas id="detection-canvas-${this.cameraId}" class="detection-canvas"></canvas>
                    <div class="detection-overlay">
                        <div id="detection-status-${this.cameraId}" class="status-indicator">
                            <span class="status-dot"></span>
                            <span class="status-text">Connecting...</span>
                        </div>
                    </div>
                </div>
                <div class="detection-controls">
                    <button id="start-btn-${this.cameraId}" class="btn btn-success" disabled>
                        <i class="fas fa-play"></i> Start Detection
                    </button>
                    <button id="stop-btn-${this.cameraId}" class="btn btn-danger" disabled>
                        <i class="fas fa-stop"></i> Stop Detection
                    </button>
                    <select id="detection-type-${this.cameraId}" class="form-select">
                        <option value="security_intrusion">Security Intrusion</option>
                        <option value="ppe_detection">PPE Detection</option>
                        <option value="object_detection">Object Detection</option>
                    </select>
                </div>
                <div id="detection-info-${this.cameraId}" class="detection-info">
                    <div class="row">
                        <div class="col">
                            <small class="text-muted">Detections: <span id="detection-count-${this.cameraId}">0</span></small>
                        </div>
                        <div class="col">
                            <small class="text-muted">FPS: <span id="fps-count-${this.cameraId}">0</span></small>
                        </div>
                        <div class="col">
                            <small class="text-muted">Latency: <span id="latency-${this.cameraId}">0ms</span></small>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Get references to UI elements
        this.canvas = document.getElementById(`detection-canvas-${this.cameraId}`);
        this.ctx = this.canvas.getContext('2d');
        this.statusElement = document.getElementById(`detection-status-${this.cameraId}`);
        this.startBtn = document.getElementById(`start-btn-${this.cameraId}`);
        this.stopBtn = document.getElementById(`stop-btn-${this.cameraId}`);
        this.detectionTypeSelect = document.getElementById(`detection-type-${this.cameraId}`);
        this.detectionCount = document.getElementById(`detection-count-${this.cameraId}`);
        this.fpsCount = document.getElementById(`fps-count-${this.cameraId}`);
        this.latencyElement = document.getElementById(`latency-${this.cameraId}`);

        // Set canvas size
        this.canvas.width = 640;
        this.canvas.height = 480;

        // Bind event listeners
        this.bindEvents();
    }

    bindEvents() {
        this.startBtn.addEventListener('click', () => this.startDetection());
        this.stopBtn.addEventListener('click', () => this.stopDetection());
        this.detectionTypeSelect.addEventListener('change', (e) => {
            this.detectionType = e.target.value;
        });
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/detection/${this.cameraId}/`;
        
        this.updateStatus('Connecting...', 'connecting');
        
        try {
            this.socket = new WebSocket(wsUrl);
            this.bindSocketEvents();
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.updateStatus('Connection Failed', 'error');
            this.scheduleReconnect();
        }
    }

    bindSocketEvents() {
        this.socket.onopen = (event) => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.updateStatus('Connected', 'connected');
            this.startBtn.disabled = false;
            this.getStatus();
        };

        this.socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };

        this.socket.onclose = (event) => {
            console.log('WebSocket disconnected:', event.code, event.reason);
            this.isConnected = false;
            this.updateStatus('Disconnected', 'disconnected');
            this.startBtn.disabled = true;
            this.stopBtn.disabled = true;
            
            if (event.code !== 1000) {
                this.scheduleReconnect();
            }
        };

        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('Connection Error', 'error');
        };
    }

    handleMessage(data) {
        const messageStartTime = performance.now();
        
        switch (data.type) {
            case 'frame':
                this.handleFrame(data);
                break;
            case 'status':
                this.handleStatus(data);
                break;
            case 'stream_started':
                this.handleStreamStarted(data);
                break;
            case 'stream_stopped':
                this.handleStreamStopped(data);
                break;
            case 'error':
                this.handleError(data);
                break;
            default:
                console.log('Unknown message type:', data);
        }
        
        // Calculate and display latency
        const latency = performance.now() - messageStartTime;
        this.latencyElement.textContent = `${latency.toFixed(1)}ms`;
    }

    handleFrame(data) {
        if (data.data) {
            const img = new Image();
            img.onload = () => {
                // Clear canvas
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                
                // Draw frame
                this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
                
                // Draw detections
                if (data.detections && data.detections.length > 0) {
                    this.drawDetections(data.detections);
                    this.updateDetectionCount(data.detections.length);
                }
                
                // Update FPS
                this.updateFPS();
            };
            img.src = `data:image/jpeg;base64,${data.data}`;
        }
    }

    drawDetections(detections) {
        this.ctx.strokeStyle = '#00ff00';
        this.ctx.lineWidth = 2;
        this.ctx.fillStyle = '#00ff00';
        this.ctx.font = '14px Arial';

        detections.forEach(detection => {
            if (detection.bbox) {
                const [x, y, w, h] = detection.bbox;
                
                // Scale coordinates to canvas size
                const scaleX = this.canvas.width / (detection.frame_width || 640);
                const scaleY = this.canvas.height / (detection.frame_height || 480);
                
                const scaledX = x * scaleX;
                const scaledY = y * scaleY;
                const scaledW = w * scaleX;
                const scaledH = h * scaleY;
                
                // Draw bounding box
                this.ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
                
                // Draw label
                const label = `${detection.class_name || 'Detection'} (${Math.round((detection.confidence || 0) * 100)}%)`;
                const textWidth = this.ctx.measureText(label).width;
                
                this.ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
                this.ctx.fillRect(scaledX, scaledY - 25, textWidth + 10, 20);
                
                this.ctx.fillStyle = '#000000';
                this.ctx.fillText(label, scaledX + 5, scaledY - 10);
            }
        });
    }

    handleStatus(data) {
        console.log('Status update:', data);
        this.updateStatus(
            data.streaming ? 'Streaming' : 'Ready',
            data.streaming ? 'streaming' : 'connected'
        );
    }

    handleStreamStarted(data) {
        console.log('Stream started:', data);
        this.updateStatus('Streaming', 'streaming');
        this.startBtn.disabled = true;
        this.stopBtn.disabled = false;
        this.detectionTypeSelect.disabled = true;
    }

    handleStreamStopped(data) {
        console.log('Stream stopped:', data);
        this.updateStatus('Ready', 'connected');
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.detectionTypeSelect.disabled = false;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.updateDetectionCount(0);
    }

    handleError(data) {
        console.error('WebSocket error:', data.message);
        this.updateStatus(`Error: ${data.message}`, 'error');
        
        // Show error notification
        if (typeof showNotification === 'function') {
            showNotification(data.message, 'error');
        } else {
            alert(`Detection Error: ${data.message}`);
        }
    }

    startDetection() {
        if (this.isConnected) {
            this.sendMessage({
                command: 'start_detection',
                type: this.detectionType
            });
        }
    }

    stopDetection() {
        if (this.isConnected) {
            this.sendMessage({
                command: 'stop_detection'
            });
        }
    }

    getStatus() {
        if (this.isConnected) {
            this.sendMessage({
                command: 'get_status'
            });
        }
    }

    sendMessage(message) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(message));
        }
    }

    updateStatus(text, status) {
        if (this.statusElement) {
            const statusText = this.statusElement.querySelector('.status-text');
            const statusDot = this.statusElement.querySelector('.status-dot');
            
            if (statusText) statusText.textContent = text;
            if (statusDot) {
                statusDot.className = 'status-dot';
                statusDot.classList.add(`status-${status}`);
            }
        }
    }

    updateDetectionCount(count) {
        if (this.detectionCount) {
            this.detectionCount.textContent = count;
        }
    }

    updateFPS() {
        // Simple FPS calculation
        const now = performance.now();
        if (!this.lastFrameTime) {
            this.lastFrameTime = now;
            this.frameCount = 0;
            return;
        }
        
        this.frameCount++;
        const elapsed = now - this.lastFrameTime;
        
        if (elapsed >= 1000) { // Update FPS every second
            const fps = Math.round((this.frameCount * 1000) / elapsed);
            if (this.fpsCount) {
                this.fpsCount.textContent = fps;
            }
            this.lastFrameTime = now;
            this.frameCount = 0;
        }
    }

    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
            
            this.updateStatus(`Reconnecting in ${Math.ceil(delay / 1000)}s (${this.reconnectAttempts}/${this.maxReconnectAttempts})`, 'reconnecting');
            
            setTimeout(() => {
                this.connect();
            }, delay);
        } else {
            this.updateStatus('Connection Failed', 'error');
        }
    }

    disconnect() {
        if (this.socket) {
            this.socket.close(1000, 'Manual disconnect');
        }
    }

    destroy() {
        this.disconnect();
        if (this.container) {
            this.container.innerHTML = '';
        }
    }
}

// CSS styles (add to your stylesheet)
const detectionStyles = `
.detection-stream-container {
    border: 1px solid #ddd;
    border-radius: 8px;
    overflow: hidden;
    background: #000;
}

.video-container {
    position: relative;
    background: #000;
}

.detection-canvas {
    width: 100%;
    height: auto;
    display: block;
}

.detection-overlay {
    position: absolute;
    top: 10px;
    left: 10px;
    z-index: 10;
}

.status-indicator {
    background: rgba(0, 0, 0, 0.7);
    color: white;
    padding: 5px 10px;
    border-radius: 4px;
    font-size: 12px;
    display: flex;
    align-items: center;
    gap: 5px;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #666;
}

.status-dot.status-connecting {
    background: #ffa500;
    animation: pulse 1s infinite;
}

.status-dot.status-connected {
    background: #28a745;
}

.status-dot.status-streaming {
    background: #007bff;
    animation: pulse 0.5s infinite;
}

.status-dot.status-disconnected {
    background: #6c757d;
}

.status-dot.status-error {
    background: #dc3545;
}

.status-dot.status-reconnecting {
    background: #ffc107;
    animation: pulse 1s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.detection-controls {
    padding: 10px;
    background: #f8f9fa;
    border-top: 1px solid #ddd;
    display: flex;
    gap: 10px;
    align-items: center;
}

.detection-info {
    padding: 5px 10px;
    background: #e9ecef;
    border-top: 1px solid #ddd;
    font-size: 11px;
}
`;

// Add styles to document
if (!document.getElementById('detection-websocket-styles')) {
    const styleSheet = document.createElement('style');
    styleSheet.id = 'detection-websocket-styles';
    styleSheet.textContent = detectionStyles;
    document.head.appendChild(styleSheet);
}

// Export for use
window.DetectionWebSocket = DetectionWebSocket;