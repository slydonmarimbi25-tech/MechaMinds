#!/usr/bin/env python3
"""
Non-Contact Heart Rate Monitor using rPPG (remote Photoplethysmography)
Flask-backed web application for GitHub Codespaces / Browser access.
"""

import cv2
import numpy as np
import scipy.signal as signal
import base64
import os
import urllib.request
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
BUFFER_SIZE = 150       # Number of frames (~5 seconds of data at 30 FPS)
DEFAULT_FPS = 30        # Target frame rate
MIN_HR_BPM = 45.0       # Minimum physiological Heart Rate
MAX_HR_BPM = 210.0      # Maximum physiological Heart Rate

# Global buffer to keep track of green channel values across incoming browser requests
green_buffer = []
current_bpm = 0.0

# ============================================================================
# FACE DETECTION
# ============================================================================
def load_cascade():
    """Download and load Haar Cascade face detector"""
    filename = "haarcascade_frontalface_default.xml"
    if not os.path.exists(filename):
        print("Downloading face detector model...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/" + filename
        urllib.request.urlretrieve(url, filename)
    return cv2.CascadeClassifier(filename)

face_cascade = load_cascade()

# ============================================================================
# SIGNAL PROCESSING ENGINE
# ============================================================================
def build_bandpass_filter(fps, low_bpm=MIN_HR_BPM, high_bpm=MAX_HR_BPM, order=3):
    """Create a Butterworth bandpass filter for heart rate frequencies"""
    low = low_bpm / 60.0
    high = high_bpm / 60.0
    nyquist = 0.5 * fps
    
    low_norm = max(0.01, min(low / nyquist, 0.98))
    high_norm = max(low_norm + 0.01, min(high / nyquist, 0.99))
    
    b, a = signal.butter(order, [low_norm, high_norm], btype='band')
    return b, a

def extract_bpm_from_signal(green_signal, fps):
    """Extract heart rate from green channel signal using FFT"""
    N = len(green_signal)
    if N < BUFFER_SIZE:
        return 0.0
    
    signal_std = np.std(green_signal)
    if signal_std < 1e-6:
        return 0.0
        
    normalized_signal = (green_signal - np.mean(green_signal)) / signal_std
    
    try:
        b, a = build_bandpass_filter(fps)
        filtered_signal = signal.filtfilt(b, a, normalized_signal)
    except Exception:
        return 0.0
    
    fft_vals = np.abs(np.fft.rfft(filtered_signal))
    fft_freqs = np.fft.rfftfreq(N, d=1.0/fps)
    
    valid_idx = np.where((fft_freqs >= MIN_HR_BPM / 60.0) & (fft_freqs <= MAX_HR_BPM / 60.0))
    valid_freqs = fft_freqs[valid_idx]
    valid_fft = fft_vals[valid_idx]
    
    if len(valid_fft) == 0:
        return 0.0
    
    peak_freq = valid_freqs[np.argmax(valid_fft)]
    return peak_freq * 60.0

# ============================================================================
# HTML & JAVASCRIPT FRONTEND (RUNS IN YOUR LOCAL BROWSER)
# ============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>rPPG Heart Rate Monitor</title>
    <style>
        body { font-family: sans-serif; text-align: center; background-color: #121212; color: #fff; padding-top: 20px; }
        #videoElement { display: none; }
        #outputImage { border: 2px solid #333; border-radius: 8px; max-width: 640px; width: 100%; height: auto; }
        .container { max-width: 700px; margin: 0 auto; }
    </style>
</head>
<body>
    <div class="container">
        <h2>rPPG Heart Rate Monitor (Codespaces Bridge)</h2>
        <p>Allow camera access if prompted. Keep this tab active.</p>
        
        <video id="videoElement" autoplay playsinline></video>
        <canvas id="canvasElement" width="640" height="480" style="display:none;"></canvas>
        <img id="outputImage" src="" alt="Live Feed Processing..." />
    </div>

    <script>
        const video = document.getElementById('videoElement');
        const canvas = document.getElementById('canvasElement');
        const ctx = canvas.getContext('2d');
        const outputImage = document.getElementById('outputImage');

        async function setupCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    sendFrame();
                };
            } catch (err) {
                alert("Could not access webcam. Please ensure browser permissions are allowed.");
                console.error(err);
            }
        }

        async function sendFrame() {
            ctx.drawImage(video, 0, 0, 640, 480);
            const dataUrl = canvas.toDataURL('image/jpeg', 0.7);

            try {
                const response = await fetch('/process_frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: dataUrl })
                });

                const result = await response.json();
                if (result.image) {
                    outputImage.src = result.image;
                }
            } catch (err) {
                console.error("Frame processing error:", err);
            }

            // Loop smoothly at target FPS
            setTimeout(sendFrame, 33);
        }

        setupCamera();
    </script>
</body>
</html>
"""

# ============================================================================
# FLASK SERVER ROUTES
# ============================================================================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global green_buffer, current_bpm

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data received'}), 400

    # Decode base64 image from browser
    header, encoded = data['image'].split(',', 1)
    binary = base64.b64decode(encoded)
    img_array = np.frombuffer(binary, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Failed to decode image'}), 400

    frame = cv2.flip(frame, 1)
    h_img, w_img, _ = frame.shape

    # Detect Face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    roi_x, roi_y, roi_w, roi_h = 0, 0, 0, 0
    face_detected = False

    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (x, y, w, h) = faces[0]

        # Forehead ROI calculation
        roi_x = max(0, x + int(w * 0.25))
        roi_y = max(0, y + int(h * 0.08))
        roi_w = min(w_img - roi_x, int(w * 0.50))
        roi_h = min(h_img - roi_y, int(h * 0.20))

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 128, 0), 1)
        face_detected = True

    # Center-Crop Fallback if no face detected
    if not face_detected:
        roi_w = int(w_img * 0.20)
        roi_h = int(h_img * 0.15)
        roi_x = (w_img - roi_w) // 2
        roi_y = (h_img - roi_y) // 3

    # Extract Green Channel Mean
    if roi_w > 0 and roi_h > 0:
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        green_val = np.mean(roi[:, :, 1])
        green_buffer.append(green_val)

        if len(green_buffer) > BUFFER_SIZE:
            green_buffer.pop(0)

        if len(green_buffer) == BUFFER_SIZE:
            current_bpm = extract_bpm_from_signal(np.array(green_buffer), DEFAULT_FPS)

        box_color = (0, 255, 0) if face_detected else (0, 255, 255)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), box_color, 2)

    # On-Screen HUD Overlay
    cv2.putText(frame, f"Heart Rate: {current_bpm:.1f} BPM", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    buffer_pct = int((len(green_buffer) / BUFFER_SIZE) * 100)
    status_str = "Status: Locked" if buffer_pct == 100 else f"Status: Buffering ({buffer_pct}%)"
    cv2.putText(frame, status_str, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Encode updated frame back to base64 JPEG
    _, jpeg = cv2.imencode('.jpg', frame)
    b64_output = base64.b64encode(jpeg).decode('utf-8')

    return jsonify({'image': f'data:image/jpeg;base64,{b64_output}'})

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # Host on 0.0.0.0 so Codespaces port forwarding picks it up
    app.run(host='0.0.0.0', port=5000, debug=False)



