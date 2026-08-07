"""Live Color Magnification - Headless Mode for Codespace"""

# ===== Imports =====
import numpy as np 
import cv2
import scipy.signal as signal
from collections import deque
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ===== Configuration =====
# Video file settings
VIDEO_PATH = "videos/face.mp4"  # Change to your video path
OUTPUT_VIDEO_PATH = "evm_output.mp4"  # Output video path

# Processing settings
RESIZE_FACTOR = 0.5  # Reduce resolution for faster processing
FRAME_BUFFER_SIZE = 150  # Number of frames to keep for filtering

# EVM Parameters
ALPHA = 30.0  # Magnification factor
LEVEL = 3  # Gaussian pyramid level
f_lo = 50/60  # 0.833 Hz (lower bound)
f_hi = 60/60  # 1.0 Hz (upper bound)

# ===== Color Space Functions =====
def rgb2yiq(rgb):
    """Converts RGB to YIQ color space"""
    y = rgb @ np.array([[0.30], [0.59], [0.11]])
    rby = rgb[:, :, (0,2)] - y
    i = np.sum(rby * np.array([[[0.74, -0.27]]]), axis=-1)
    q = np.sum(rby * np.array([[[0.48, 0.41]]]), axis=-1)
    yiq = np.dstack((y.squeeze(), i, q))
    return yiq

def bgr2yiq(bgr):
    """Converts BGR to YIQ"""
    rgb = np.float32(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return rgb2yiq(rgb)

def yiq2rgb(yiq):
    """Converts YIQ to RGB"""
    r = yiq @ np.array([1.0, 0.9468822170900693, 0.6235565819861433])
    g = yiq @ np.array([1.0, -0.27478764629897834, -0.6356910791873801])
    b = yiq @ np.array([1.0, -1.1085450346420322, 1.7090069284064666])
    rgb = np.clip(np.dstack((r, g, b)), 0, 1)
    return rgb

def inv_colorspace(x):
    """Convert YIQ back to BGR"""
    return cv2.normalize(yiq2rgb(x), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)

def gaussian_pyramid(image, level):
    """Builds Gaussian pyramid"""
    rows, cols, colors = image.shape
    scale = 2**level
    pyramid = np.zeros((colors, rows//scale, cols//scale))
    
    img = image.copy()
    for i in range(level):
        img = cv2.pyrDown(img)
        rows, cols, _ = img.shape
        if i == (level - 1):
            for c in range(colors):
                pyramid[c, :, :] = img[:, :, c]
    
    return pyramid

# ===== EVM Processor =====
class EVMProcessor:
    def __init__(self, buffer_size=FRAME_BUFFER_SIZE, alpha=ALPHA, level=LEVEL):
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.level = level
        self.frame_buffer = deque(maxlen=buffer_size)
        self.yiq_buffer = deque(maxlen=buffer_size)
        self.pyramid_buffer = deque(maxlen=buffer_size)
        self.filtered_buffer = deque(maxlen=buffer_size)
        self.original_frames = deque(maxlen=buffer_size)
        
        self.fs = 30.0
        self.bandpass = None
        self.transfer_function = None
        
        self.rows = None
        self.cols = None
        self.colors = 3
        
        self.bpm = 0
        self.heart_rates = deque(maxlen=10)
        self.processed_frames = []
    
    def create_filter(self, num_taps):
        """Create bandpass filter"""
        try:
            self.bandpass = signal.firwin(
                numtaps=num_taps,
                cutoff=(f_lo, f_hi),
                fs=self.fs,
                pass_zero=False
            )
            self.transfer_function = np.fft.fft(np.fft.ifftshift(self.bandpass))
            return True
        except:
            try:
                num_taps = min(num_taps, 50)
                self.bandpass = signal.firwin(
                    numtaps=num_taps,
                    cutoff=(0.8, 1.2),
                    fs=self.fs,
                    pass_zero=False
                )
                self.transfer_function = np.fft.fft(np.fft.ifftshift(self.bandpass))
                return True
            except:
                return False
    
    def process_frame(self, frame):
        """Process a single frame through EVM pipeline"""
        # Resize frame
        if RESIZE_FACTOR < 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * RESIZE_FACTOR)
            new_h = int(h * RESIZE_FACTOR)
            frame_resized = cv2.resize(frame, (new_w, new_h))
        else:
            frame_resized = frame.copy()
        
        # Store original dimensions
        if self.rows is None:
            self.rows, self.cols = frame_resized.shape[:2]
        
        # Store original frame
        self.original_frames.append(frame_resized)
        
        # Convert to YIQ
        yiq_frame = bgr2yiq(np.float32(frame_resized/255))
        self.yiq_buffer.append(yiq_frame)
        
        # Build pyramid
        pyramid = gaussian_pyramid(yiq_frame, self.level)
        self.pyramid_buffer.append(pyramid)
        
        # Process if buffer is full
        result_frame = None
        if len(self.pyramid_buffer) >= self.buffer_size:
            # Initialize filter if needed
            if self.bandpass is None:
                self.create_filter(self.buffer_size)
            
            # Apply temporal filtering
            if self.bandpass is not None:
                pyramid_stack = np.array(self.pyramid_buffer)
                
                # Apply FFT filtering
                pyr_fft = np.fft.fft(pyramid_stack, axis=0).astype(np.complex64)
                
                # Apply filter
                tf = self.transfer_function[:len(pyramid_stack)]
                if len(tf) < len(pyramid_stack):
                    tf_padded = np.zeros(len(pyramid_stack), dtype=tf.dtype)
                    tf_padded[:len(tf)] = tf
                else:
                    tf_padded = tf[:len(pyramid_stack)]
                
                filtered = pyr_fft * tf_padded[:, None, None, None].astype(np.complex64)
                filtered_pyramid = np.fft.ifft(filtered, axis=0).real
                
                # Get the most recent filtered pyramid
                current_filtered = filtered_pyramid[-1]
                
                # Extract pyramid channels
                y_chan = pyramid_stack[-1, 0, :, :]
                i_chan = pyramid_stack[-1, 1, :, :]
                q_chan = pyramid_stack[-1, 2, :, :]
                
                # Get filtered channels
                fy_chan = current_filtered[0, :, :] * self.alpha
                fi_chan = current_filtered[1, :, :] * self.alpha
                fq_chan = current_filtered[2, :, :] * self.alpha
                
                # Upscale filtered channels
                fy_chan = cv2.resize(fy_chan, (self.cols, self.rows))
                fi_chan = cv2.resize(fi_chan, (self.cols, self.rows))
                fq_chan = cv2.resize(fq_chan, (self.cols, self.rows))
                
                # Upscale original pyramid channels
                y_chan_up = cv2.resize(y_chan, (self.cols, self.rows))
                i_chan_up = cv2.resize(i_chan, (self.cols, self.rows))
                q_chan_up = cv2.resize(q_chan, (self.cols, self.rows))
                
                # Create magnified YIQ
                yiq_mag = np.zeros((self.rows, self.cols, 3), dtype=np.float32)
                yiq_mag[:, :, 0] = y_chan_up + fy_chan
                yiq_mag[:, :, 1] = i_chan_up + fi_chan
                yiq_mag[:, :, 2] = q_chan_up + fq_chan
                
                # Convert to RGB
                result_frame = inv_colorspace(yiq_mag)
                
                # Store filtered buffer for heart rate detection
                self.filtered_buffer.append(result_frame)
                
                # Detect heart rate periodically
                if len(self.filtered_buffer) % 30 == 0:
                    self.detect_heart_rate()
        
        # Return original frame if buffer not full yet
        if result_frame is None:
            result_frame = frame_resized
        
        return result_frame
    
    def detect_heart_rate(self):
        """Detect heart rate from filtered frames"""
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
        except:
            pass

# ===== Main Function (Headless) =====
def main():
    print("="*70)
    print("COLOR MAGNIFICATION - Headless Mode (Codespace Compatible)")
    print("="*70)
    print(f"Video path: {VIDEO_PATH}")
    print(f"Buffer size: {FRAME_BUFFER_SIZE} frames")
    print(f"Magnification factor: {ALPHA}x")
    print(f"Processing resolution: {RESIZE_FACTOR*100:.0f}%")
    print("="*70)
    
    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: Video file not found: {VIDEO_PATH}")
        print("Please update VIDEO_PATH to your video file location")
        return
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {VIDEO_PATH}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ Video loaded: {VIDEO_PATH}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Total frames: {total_frames}")
    
    # Initialize processor
    processor = EVMProcessor()
    processor.fs = fps
    
    print("\n🔄 Processing video...")
    print("This may take a few minutes...")
    print("-"*70)
    
    # Process all frames
    frame_count = 0
    processed_frames_original = []
    processed_frames_magnified = []
    
    start_time = time.time()
    last_progress = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame
        processed_frame = processor.process_frame(frame)
        
        # Store frames for output
        processed_frames_original.append(frame)
        processed_frames_magnified.append(processed_frame)
        
        # Show progress
        progress = int((frame_count / total_frames) * 100)
        if progress >= last_progress + 10:
            elapsed = time.time() - start_time
            estimated_total = (elapsed / frame_count) * total_frames
            remaining = estimated_total - elapsed
            
            print(f"  Progress: {progress}% ({frame_count}/{total_frames}) | "
                  f"Buffer: {len(processor.pyramid_buffer)}/{processor.buffer_size} | "
                  f"BPM: {processor.bpm:.1f} | "
                  f"Remaining: {remaining:.1f}s")
            last_progress = progress
    
    cap.release()
    print("✅ Processing complete!")
    
    # Print summary
    print("\n" + "="*70)
    print("PROCESSING SUMMARY")
    print("="*70)
    print(f"Frames processed: {frame_count}")
    if processor.bpm > 0:
        print(f"✅ Estimated Heart Rate: {processor.bpm:.1f} BPM")
    else:
        print("⚠️ Heart rate not detected")
    print(f"Buffer size: {len(processor.pyramid_buffer)} frames")
    print("="*70)
    
    # Create output video (side-by-side)
    print("\n📹 Creating output video...")
    
    # Resize for output
    display_width = 640
    display_height = 480
    
    # Determine output dimensions
    h, w = processed_frames_original[0].shape[:2]
    aspect_ratio = w / h
    
    if h > display_height:
        h = display_height
        w = int(h * aspect_ratio)
    if w > display_width:
        w = display_width
        h = int(w / aspect_ratio)
    
    # Ensure dimensions are even
    if h % 2 != 0:
        h += 1
    if w % 2 != 0:
        w += 1
    
    # Create video writer
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, int(fps), (w*2 + 5, h))
        
        for i in range(len(processed_frames_original)):
            # Resize frames
            orig = cv2.resize(processed_frames_original[i], (w, h))
            mag = cv2.resize(processed_frames_magnified[i], (w, h))
            
            # Convert to BGR if needed
            if len(orig.shape) == 3 and orig.shape[2] == 3:
                orig = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
            if len(mag.shape) == 3 and mag.shape[2] == 3:
                mag = cv2.cvtColor(mag, cv2.COLOR_RGB2BGR)
            
            # Add labels
            cv2.putText(orig, "ORIGINAL", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(mag, f"MAGNIFIED (BPM: {processor.bpm:.1f})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add frame counter
            cv2.putText(orig, f"Frame: {i+1}/{len(processed_frames_original)}", 
                       (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Combine side by side
            separator = np.ones((h, 5, 3), dtype=np.uint8) * 255
            combined = np.hstack([orig, separator, mag])
            
            out.write(combined)
        
        out.release()
        print(f"✅ Output video saved: {OUTPUT_VIDEO_PATH}")
        print(f"   Resolution: {w*2+5}x{h}")
        print(f"   Frames: {len(processed_frames_original)}")
        
    except Exception as e:
        print(f"❌ Error creating video: {e}")
    
    # Also create a GIF
    print("\n🎞️ Creating GIF preview...")
    try:
        from PIL import Image
        
        gif_frames = []
        step = max(1, len(processed_frames_original) // 30)  # Use ~30 frames for GIF
        
        for i in range(0, len(processed_frames_original), step):
            # Get frame
            frame = processed_frames_magnified[i]
            
            # Convert to RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Resize for GIF
            gif_h, gif_w = frame_rgb.shape[:2]
            max_size = 400
            if gif_w > max_size:
                scale = max_size / gif_w
                new_w = int(gif_w * scale)
                new_h = int(gif_h * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
            
            gif_frames.append(Image.fromarray(frame_rgb))
        
        if gif_frames:
            gif_path = "evm_output.gif"
            gif_frames[0].save(
                gif_path,
                format="GIF",
                append_images=gif_frames[1:],
                save_all=True,
                duration=100,
                loop=0
            )
            print(f"✅ GIF saved: {gif_path}")
    
    except Exception as e:
        print(f"⚠️ Could not create GIF: {e}")
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print(f"Output files:")
    print(f"  - Video: {OUTPUT_VIDEO_PATH}")
    print(f"  - GIF: evm_output.gif")
    print("="*70)

# ===== Run =====
if __name__ == "__main__":
    main()