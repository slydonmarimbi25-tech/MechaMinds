"""Converted from color_mag.ipynb - Optimized for Codespace"""

# ===== Imports =====
import os
import numpy as np 
import cv2
import matplotlib
matplotlib.use('Agg')  # Headless backend for Codespace
import matplotlib.pyplot as plt
import scipy.signal as signal
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ===== Configuration - REDUCED FOR CODESPACE =====
DATA_PATH = "videos"
VIDEO_NAME = "face.mp4"
VIDEO_PATH = os.path.join(DATA_PATH, VIDEO_NAME)

# Memory-saving settings
SCALE_FACTOR = 0.3  # Process at 30% resolution
LEVEL = 3  # Reduced pyramid level (was 4)
MAX_FRAMES = 150  # Process fewer frames
ALPHA = 50.0  # Magnification factor

# Temporal filter parameters (for heart rate ~60-100 BPM)
f_lo = 50/60  # 0.833 Hz
f_hi = 60/60  # 1.0 Hz

print(f"VIDEO_PATH: {VIDEO_PATH}")
print(f"Exists: {os.path.exists(VIDEO_PATH)}")

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
    
    # Make sure we downsample the correct number of times
    img = image.copy()
    for i in range(level):
        img = cv2.pyrDown(img)
        rows, cols, _ = img.shape
        if i == (level - 1):
            for c in range(colors):
                pyramid[c, :, :] = img[:, :, c]
    
    return pyramid

# ===== Load Video Frames =====
print("\nLoading video frames...")
frames = []
cap = cv2.VideoCapture(VIDEO_PATH)
fs = cap.get(cv2.CAP_PROP_FPS)

if fs <= 0:
    fs = 30.0  # Default if detection fails

idx = 0

while(cap.isOpened() and idx < MAX_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    
    if idx == 0:
        og_h, og_w, _ = frame.shape
        w = int(og_w * SCALE_FACTOR)
        h = int(og_h * SCALE_FACTOR)
        print(f"Original size: {og_w}x{og_h}, Processing at: {w}x{h}")
    
    # Convert to YIQ
    frame = bgr2yiq(np.float32(frame/255))
    
    # Resize
    if SCALE_FACTOR < 1.0:
        frame = cv2.resize(frame, (w, h))
    
    frames.append(frame)
    idx += 1

cap.release()
cv2.destroyAllWindows()

NUM_FRAMES = len(frames)
print(f"✅ Loaded {NUM_FRAMES} frames at {fs} FPS")

if NUM_FRAMES == 0:
    print("❌ No frames loaded! Check video path.")
    exit()

# ===== Create Temporal Filter =====
print("\nCreating temporal filter...")
# IMPORTANT: Use NUM_FRAMES for filter taps, not a fixed number
numtaps = NUM_FRAMES  # Use all frames for better filtering

try:
    bandpass = signal.firwin(
        numtaps=numtaps,
        cutoff=(f_lo, f_hi),
        fs=fs,
        pass_zero=False
    )
    print(f"✅ Filter created with {numtaps} taps")
except Exception as e:
    print(f"⚠️ Filter creation failed: {e}")
    # Fallback to simpler filter with fewer taps
    numtaps = min(NUM_FRAMES, 50)
    bandpass = signal.firwin(
        numtaps=numtaps,
        cutoff=(0.8, 1.2),
        fs=fs,
        pass_zero=False
    )
    print(f"✅ Fallback filter created with {numtaps} taps")

transfer_function = np.fft.fft(np.fft.ifftshift(bandpass))

# ===== Build Gaussian Pyramid =====
print("\nBuilding Gaussian pyramid...")
rows, cols, colors = frames[0].shape
scale = 2**LEVEL

# Ensure dimensions are divisible by scale
rows_proc = (rows // scale) * scale
cols_proc = (cols // scale) * scale

pyramid_stack = np.zeros((NUM_FRAMES, colors, rows_proc//scale, cols_proc//scale), dtype=np.float32)

for i, frame in enumerate(frames):
    # Crop to divisible size if needed
    if frame.shape[0] != rows_proc or frame.shape[1] != cols_proc:
        frame = frame[:rows_proc, :cols_proc, :]
    
    pyramid = gaussian_pyramid(frame, LEVEL)
    pyramid_stack[i, :, :, :] = pyramid

print(f"✅ Pyramid built: {pyramid_stack.shape}")

# Free some memory
del frames

# ===== Apply Temporal Filtering =====
print("\nApplying temporal filtering...")
try:
    # FFT along time axis
    pyr_stack_fft = np.fft.fft(pyramid_stack, axis=0).astype(np.complex64)
    
    # IMPORTANT FIX: Ensure transfer_function matches the number of frames
    # If transfer_function is shorter, pad it; if longer, truncate
    if len(transfer_function) < NUM_FRAMES:
        # Pad with zeros
        tf_padded = np.zeros(NUM_FRAMES, dtype=transfer_function.dtype)
        tf_padded[:len(transfer_function)] = transfer_function
    else:
        # Truncate to NUM_FRAMES
        tf_padded = transfer_function[:NUM_FRAMES]
    
    # Apply filter with correct broadcasting
    _filtered_pyramid = pyr_stack_fft * tf_padded[:, None, None, None].astype(np.complex64)
    
    # Inverse FFT
    filtered_pyramid = np.fft.ifft(_filtered_pyramid, axis=0).real
    
    # Free memory
    del pyr_stack_fft, _filtered_pyramid, tf_padded
    
    print(f"✅ Filtering complete: {filtered_pyramid.shape}")
    
except MemoryError as e:
    print(f"❌ Memory error during filtering: {e}")
    print("Try reducing SCALE_FACTOR or MAX_FRAMES further")
    exit()
except Exception as e:
    print(f"❌ Error during filtering: {e}")
    exit()

# ===== Apply Magnification =====
print("\nApplying magnification...")
magnified = []
magnified_only = []

# Store original pyramid for later use (we already have pyramid_stack)
original_pyramid = pyramid_stack.copy()

for i in range(NUM_FRAMES):
    try:
        # Get original channels from pyramid
        y_chan = original_pyramid[i, 0, :, :]
        i_chan = original_pyramid[i, 1, :, :]
        q_chan = original_pyramid[i, 2, :, :]
        
        # Get filtered channels
        fy_chan = filtered_pyramid[i, 0, :, :] * ALPHA
        fi_chan = filtered_pyramid[i, 1, :, :] * ALPHA
        fq_chan = filtered_pyramid[i, 2, :, :] * ALPHA
        
        # Upscale filtered channels back to original size
        fy_chan = cv2.resize(fy_chan, (cols_proc, rows_proc))
        fi_chan = cv2.resize(fi_chan, (cols_proc, rows_proc))
        fq_chan = cv2.resize(fq_chan, (cols_proc, rows_proc))
        
        # Upscale original pyramid channels too
        y_chan_up = cv2.resize(y_chan, (cols_proc, rows_proc))
        i_chan_up = cv2.resize(i_chan, (cols_proc, rows_proc))
        q_chan_up = cv2.resize(q_chan, (cols_proc, rows_proc))
        
        # Create YIQ image
        yiq_mag = np.zeros((rows_proc, cols_proc, 3), dtype=np.float32)
        yiq_mag[:, :, 0] = y_chan_up + fy_chan
        yiq_mag[:, :, 1] = i_chan_up + fi_chan
        yiq_mag[:, :, 2] = q_chan_up + fq_chan
        
        # Convert to RGB
        mag = inv_colorspace(yiq_mag)
        magnified.append(mag)
        
        # Store magnified only
        mag_only = np.dstack((fy_chan, fi_chan, fq_chan))
        magnified_only.append(mag_only)
        
    except Exception as e:
        print(f"⚠️ Error processing frame {i}: {e}")
        continue

print(f"✅ Magnification complete: {len(magnified)} frames")

if len(magnified) == 0:
    print("❌ No frames were magnified successfully!")
    exit()

# ===== Simple Heart Rate Detection =====
print("\nDetecting heart rate...")
try:
    # Use the magnified video to detect heart rate
    reds = []
    for i in range(min(NUM_FRAMES, len(magnified))):
        if i < len(magnified):
            reds.append(np.mean(magnified[i][:, :, 0]))

    reds = np.array(reds)

    if len(reds) > 1:
        freqs = np.fft.rfftfreq(len(reds)) * fs
        rates = np.abs(np.fft.rfft(reds)) / len(reds)
        
        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(rates[1:], height=np.max(rates[1:]) * 0.3)
        
        if len(peaks) > 0:
            peak_idx = peaks[0] + 1
            bpm = freqs[peak_idx] * 60
            print(f"✅ Estimated heart rate: {bpm:.1f} BPM")
        else:
            print("⚠️ Could not detect heart rate")
    else:
        print("⚠️ Not enough frames for heart rate detection")
except Exception as e:
    print(f"⚠️ Heart rate detection failed: {e}")

# ===== Create Output Video =====
print("\nCreating output video...")
try:
    # Get dimensions
    h, w, _ = magnified[0].shape
    
    # Create stacked frames
    stacked_frames = []
    for i in range(min(NUM_FRAMES, len(magnified))):
        # Original frame from pyramid
        og_frame = np.zeros((rows_proc, cols_proc, 3), dtype=np.float32)
        og_frame[:, :, 0] = cv2.resize(original_pyramid[i, 0, :, :], (cols_proc, rows_proc))
        og_frame[:, :, 1] = cv2.resize(original_pyramid[i, 1, :, :], (cols_proc, rows_proc))
        og_frame[:, :, 2] = cv2.resize(original_pyramid[i, 2, :, :], (cols_proc, rows_proc))
        
        og_frame = inv_colorspace(og_frame)
        
        # Create side-by-side display
        middle = np.zeros((h, 5, 3), dtype=np.uint8)
        
        left = cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR)
        right = cv2.cvtColor(magnified[i], cv2.COLOR_RGB2BGR)
        
        # Resize to match if needed
        if left.shape[0] != right.shape[0] or left.shape[1] != right.shape[1]:
            right = cv2.resize(right, (left.shape[1], left.shape[0]))
        
        combined = np.hstack([left, middle, right])
        stacked_frames.append(combined)
    
    if len(stacked_frames) > 0:
        # Save video
        output_path = f"stacked_{int(ALPHA)}x.mp4"
        _h, _w, _ = stacked_frames[-1].shape
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'), 
            int(min(fs, 30)), 
            (_w, _h)
        )
        
        for frame in stacked_frames:
            out.write(frame)
        
        out.release()
        print(f"✅ Video saved: {output_path}")
    else:
        print("⚠️ No frames to create video")
    
except Exception as e:
    print(f"⚠️ Could not create video: {e}")

# ===== Try to create simple plot =====
print("\nGenerating visualization...")
try:
    # Plot average intensity over time
    plt.figure(figsize=(10, 4))
    times = np.arange(0, min(NUM_FRAMES, len(magnified))) / fs
    
    # Get average red channel intensity
    red_means = []
    for i in range(min(NUM_FRAMES, len(magnified))):
        if i < len(magnified):
            red_means.append(np.mean(magnified[i][:, :, 0]))
    
    if len(red_means) > 1:
        plt.plot(times[:len(red_means)], red_means, 'r-', label='Red Channel')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Intensity')
        plt.title(f'Color Magnification Signal (ALPHA={ALPHA}x)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('magnification_signal.png', dpi=150, bbox_inches='tight')
        print("✅ Plot saved: magnification_signal.png")
    
    plt.close()
    
except Exception as e:
    print(f"⚠️ Could not generate plot: {e}")

print("\n" + "="*50)
print("✅ Processing Complete!")
print("="*50)
print(f"Processed {NUM_FRAMES} frames at {fs} FPS")
print(f"Magnification factor: {ALPHA}x")
print(f"Gaussian pyramid level: {LEVEL}")
print(f"Output video: stacked_{int(ALPHA)}x.mp4")
print("="*50)

