"""Live EVM-style processing for a browser camera feed."""

import base64
import binascii
import warnings
from collections import deque

import cv2
import numpy as np
import scipy.signal as signal
from flask import Flask, jsonify, render_template_string, request

warnings.filterwarnings("ignore")

# ===== Configuration =====
RESIZE_FACTOR = 0.5
FRAME_BUFFER_SIZE = 120
ALPHA = 30.0
LEVEL = 3
f_lo = 50 / 60.0
f_hi = 60 / 60.0
DEFAULT_FPS = 30.0

app = Flask(__name__)
processor = None


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def rgb2yiq(rgb):
    """Convert RGB to YIQ color space."""
    y = rgb @ np.array([[0.30], [0.59], [0.11]])
    rby = rgb[:, :, (0, 2)] - y
    i = np.sum(rby * np.array([[[0.74, -0.27]]]), axis=-1)
    q = np.sum(rby * np.array([[[0.48, 0.41]]]), axis=-1)
    yiq = np.dstack((y.squeeze(), i, q))
    return yiq


def bgr2yiq(bgr):
    """Convert BGR to YIQ."""
    rgb = np.float32(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return rgb2yiq(rgb)


def yiq2rgb(yiq):
    """Convert YIQ to RGB."""
    r = yiq @ np.array([1.0, 0.9468822170900693, 0.6235565819861433])
    g = yiq @ np.array([1.0, -0.27478764629897834, -0.6356910791873801])
    b = yiq @ np.array([1.0, -1.1085450346420322, 1.7090069284064666])
    rgb = np.clip(np.dstack((r, g, b)), 0, 1)
    return rgb


def inv_colorspace(x):
    """Convert YIQ back to a displayable BGR frame."""
    rgb = yiq2rgb(x)
    rgb = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def gaussian_pyramid(image, level):
    """Build a Gaussian pyramid for the given image and return the final level as a channel-first array."""
    img = image.copy()
    for _ in range(max(0, level)):
        if img.shape[0] < 2 or img.shape[1] < 2:
            break
        img = cv2.pyrDown(img)

    rows, cols, colors = img.shape
    pyramid = np.zeros((colors, rows, cols), dtype=np.float32)
    for c in range(colors):
        pyramid[c, :, :] = img[:, :, c]

    return pyramid


class EVMProcessor:
    def __init__(self, buffer_size=FRAME_BUFFER_SIZE, alpha=ALPHA, level=LEVEL):
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.level = level
        self.pyramid_buffer = deque(maxlen=buffer_size)
        self.filtered_buffer = deque(maxlen=buffer_size)
        self.fs = DEFAULT_FPS
        self.bandpass = None
        self.transfer_function = None
        self.rows = None
        self.cols = None
        self.bpm = 0.0
        self.heart_rates = deque(maxlen=10)

    def create_filter(self, num_taps):
        """Create a band-pass filter for heart-rate frequencies."""
        try:
            self.bandpass = signal.firwin(
                numtaps=num_taps,
                cutoff=(f_lo, f_hi),
                fs=self.fs,
                pass_zero=False,
            )
            self.transfer_function = np.fft.fft(np.fft.ifftshift(self.bandpass))
            return True
        except Exception:
            try:
                num_taps = min(num_taps, 50)
                self.bandpass = signal.firwin(
                    numtaps=num_taps,
                    cutoff=(0.8, 1.2),
                    fs=self.fs,
                    pass_zero=False,
                )
                self.transfer_function = np.fft.fft(np.fft.ifftshift(self.bandpass))
                return True
            except Exception:
                return False

    def process_frame(self, frame):
        """Process a single frame through the EVM pipeline."""
        if frame is None:
            return None

        if RESIZE_FACTOR < 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * RESIZE_FACTOR)
            new_h = int(h * RESIZE_FACTOR)
            frame_resized = cv2.resize(frame, (new_w, new_h))
        else:
            frame_resized = frame.copy()

        if self.rows is None:
            self.rows, self.cols = frame_resized.shape[:2]

        yiq_frame = bgr2yiq(np.float32(frame_resized / 255.0))
        pyramid = gaussian_pyramid(yiq_frame, self.level)
        self.pyramid_buffer.append(pyramid)

        result_frame = frame_resized
        if len(self.pyramid_buffer) >= self.buffer_size:
            if self.bandpass is None:
                self.create_filter(self.buffer_size)

            if self.bandpass is not None:
                pyramid_stack = np.array(self.pyramid_buffer)
                pyr_fft = np.fft.fft(pyramid_stack, axis=0).astype(np.complex64)

                tf = self.transfer_function[:len(pyramid_stack)]
                if len(tf) < len(pyramid_stack):
                    tf_padded = np.zeros(len(pyramid_stack), dtype=tf.dtype)
                    tf_padded[:len(tf)] = tf
                else:
                    tf_padded = tf[:len(pyramid_stack)]

                filtered = pyr_fft * tf_padded[:, None, None, None].astype(np.complex64)
                filtered_pyramid = np.fft.ifft(filtered, axis=0).real

                current_filtered = filtered_pyramid[-1]
                y_chan = pyramid_stack[-1, 0, :, :]
                i_chan = pyramid_stack[-1, 1, :, :]
                q_chan = pyramid_stack[-1, 2, :, :]

                fy_chan = current_filtered[0, :, :] * self.alpha
                fi_chan = current_filtered[1, :, :] * self.alpha
                fq_chan = current_filtered[2, :, :] * self.alpha

                fy_chan = cv2.resize(fy_chan, (self.cols, self.rows))
                fi_chan = cv2.resize(fi_chan, (self.cols, self.rows))
                fq_chan = cv2.resize(fq_chan, (self.cols, self.rows))

                y_chan_up = cv2.resize(y_chan, (self.cols, self.rows))
                i_chan_up = cv2.resize(i_chan, (self.cols, self.rows))
                q_chan_up = cv2.resize(q_chan, (self.cols, self.rows))

                yiq_mag = np.zeros((self.rows, self.cols, 3), dtype=np.float32)
                yiq_mag[:, :, 0] = y_chan_up + fy_chan
                yiq_mag[:, :, 1] = i_chan_up + fi_chan
                yiq_mag[:, :, 2] = q_chan_up + fq_chan

                result_frame = inv_colorspace(yiq_mag)
                self.filtered_buffer.append(result_frame)

                if len(self.filtered_buffer) % 30 == 0:
                    self.detect_heart_rate()

        return result_frame

    def detect_heart_rate(self):
        """Estimate a heart rate from the magnified signal."""
        if len(self.filtered_buffer) < 30:
            return

        try:
            red_means = []
            for frame in self.filtered_buffer:
                red_means.append(np.mean(frame[:, :, 0]))

            if len(red_means) > 1:
                freqs = np.fft.rfftfreq(len(red_means)) * self.fs
                rates = np.abs(np.fft.rfft(red_means)) / len(red_means)
                peaks, _ = signal.find_peaks(rates[1:], height=np.max(rates[1:]) * 0.3)

                if len(peaks) > 0:
                    peak_idx = peaks[0] + 1
                    bpm = freqs[peak_idx] * 60
                    if 40 < bpm < 180:
                        self.heart_rates.append(bpm)
                        self.bpm = np.mean(self.heart_rates)
        except Exception:
            pass


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Live EVM Camera Feed</title>
    <style>
        body { font-family: sans-serif; text-align: center; background: #111; color: #fff; padding: 24px; }
        #videoElement { display: none; }
        #outputImage { border: 2px solid #333; border-radius: 8px; max-width: 640px; width: 100%; height: auto; }
        .container { max-width: 700px; margin: 0 auto; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Live EVM Camera Feed</h2>
        <p>Allow camera access in the browser window. The stream will be processed in real time.</p>
        <video id="videoElement" autoplay playsinline></video>
        <canvas id="canvasElement" width="640" height="480" style="display:none;"></canvas>
        <img id="outputImage" src="" alt="Processed live feed" />
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
                alert('Could not access webcam. Please allow browser camera access.');
                console.error(err);
            }
        }

        async function sendFrame() {
            ctx.drawImage(video, 0, 0, 640, 480);
            const dataUrl = canvas.toDataURL('image/jpeg', 0.75);
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
                console.error('Frame processing error:', err);
            }
            setTimeout(sendFrame, 33);
        }

        setupCamera();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/process_frame', methods=['POST', 'OPTIONS'])
def process_frame():
    if request.method == 'OPTIONS':
        return '', 200

    global processor
    if processor is None:
        processor = EVMProcessor()

    data = request.get_json(silent=True)
    if not data or 'image' not in data or not isinstance(data['image'], str):
        return jsonify({'error': 'No image data received'}), 400

    image_data = data['image']
    if ',' not in image_data or not image_data.startswith('data:image/'):
        return jsonify({'error': 'Image payload must be a data URL'}), 400

    try:
        _, encoded = image_data.split(',', 1)
        padding = '=' * (-len(encoded) % 4)
        normalized = encoded + padding
        binary = base64.b64decode(normalized, validate=True)
        img_array = np.frombuffer(binary, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except (binascii.Error, ValueError, TypeError) as exc:
        return jsonify({'error': 'Failed to decode image data'}), 400

    if frame is None:
        return jsonify({'error': 'Failed to decode image'}), 400

    frame = cv2.flip(frame, 1)
    try:
        processed_frame = processor.process_frame(frame)
    except Exception:
        processed_frame = frame

    if processed_frame is None:
        processed_frame = frame

    cv2.putText(processed_frame, f"BPM: {processor.bpm:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(processed_frame, "Live EVM Feed", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    _, jpeg = cv2.imencode('.jpg', processed_frame)
    b64_output = base64.b64encode(jpeg).decode('utf-8')
    return jsonify({'image': f'data:image/jpeg;base64,{b64_output}'})


def main():
    print("=" * 70)
    print("LIVE EVM CAMERA FEED")
    print("=" * 70)
    print("Open the forwarded port in your browser to use your webcam.")
    print("=" * 70)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
