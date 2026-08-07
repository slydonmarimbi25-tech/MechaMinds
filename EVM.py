"""Converted from color_mag.ipynb"""


# ===== Markdown Cell 1 =====
# # **Color Magnification**
# 
# In this notebook we will learn how to reveal hidden color vairations in a video! This is called color magnfiication. 

# ===== Code Cell 2 =====
import os
from glob import glob
import numpy as np 
import cv2
import matplotlib.pyplot as plt 

# Removed notebook magic: %matplotlib inline

# ===== Markdown Cell 3 =====
# Get video path

# ===== Markdown Cell 4 =====
# Add your datapath here
# You get download videos used for this tutorial [here](http://people.csail.mit.edu/mrub/evm/)

# ===== Code Cell 5 =====
#DATA_PATH = r"C:\Users\itber\Documents\learning\self_tutorials\phase_based\videos" # add your data path here
DATA_PATH = "videos"




# ===== Code Cell 6 =====
VIDEO_NAME = "face.mp4"

VIDEO_PATH = os.path.join(DATA_PATH, VIDEO_NAME)
print("VIDEO_PATH:", VIDEO_PATH)
print("Exists:", os.path.exists(VIDEO_PATH))






#os.path.exists(VIDEO_PATH)

#import os




# ===== Markdown Cell 7 =====
# ## Set Hyperparameters

# ===== Code Cell 8 =====
# video magnification factor
ALPHA = 50.0

# Gaussian Pyramid Level of which to apply magnfication
LEVEL = 4

# Temporal Filter parameters
f_lo = 50/60
f_hi = 60/60

# OPTIONAL: override fs
MANUAL_FS = None
VIDEO_FS = None

# video frame scale factor
SCALE_FACTOR = 1.0

# ===== Markdown Cell 9 =====
# ### Colorspace Functions

# ===== Code Cell 10 =====
## Color spaces
def rgb2yiq(rgb):
    """ Converts an RGB image to YIQ using FCC NTSC format.
        This is a numpy version of the colorsys implementation
        https://github.com/python/cpython/blob/main/Lib/colorsys.py
        Inputs:
            rgb - (N,M,3) rgb image
        Outputs
            yiq - (N,M,3) YIQ image
        """
    # compute Luma Channel
    y = rgb @ np.array([[0.30], [0.59], [0.11]])

    # subtract y channel from red and blue channels
    rby = rgb[:, :, (0,2)] - y

    i = np.sum(rby * np.array([[[0.74, -0.27]]]), axis=-1)
    q = np.sum(rby * np.array([[[0.48, 0.41]]]), axis=-1)

    yiq = np.dstack((y.squeeze(), i, q))
    
    return yiq


def bgr2yiq(bgr):
    """ Coverts a BGR image to float32 YIQ """
    # get normalized YIQ frame
    rgb = np.float32(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    yiq = rgb2yiq(rgb)

    return yiq


def yiq2rgb(yiq):
    """ Converts a YIQ image to RGB.
        Inputs:
            yiq - (N,M,3) YIQ image
        Outputs:
            rgb - (N,M,3) rgb image
        """
    r = yiq @ np.array([1.0, 0.9468822170900693, 0.6235565819861433])
    g = yiq @ np.array([1.0, -0.27478764629897834, -0.6356910791873801])
    b = yiq @ np.array([1.0, -1.1085450346420322, 1.7090069284064666])
    rgb = np.clip(np.dstack((r, g, b)), 0, 1)
    return rgb


inv_colorspace = lambda x: cv2.normalize(
    yiq2rgb(x), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)

# ===== Markdown Cell 11 =====
# #### Get Video Frames

# ===== Code Cell 12 =====
frames = [] # frames for processing
cap = cv2.VideoCapture(VIDEO_PATH)

# video sampling rate
fs = cap.get(cv2.CAP_PROP_FPS)

idx = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        break

    if idx == 0:
        og_h, og_w, _ = frame.shape
        w = int(og_w*SCALE_FACTOR)
        h = int(og_h*SCALE_FACTOR)

    # convert normalized uint8 BGR to the desired color space
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = bgr2yiq(np.float32(frame/255))

    # append resized frame
    frames.append(cv2.resize(frame, (w, h)))

    idx += 1
    
    
cap.release()
cv2.destroyAllWindows()
del cap

print("✅ Finished loading frames")
print("Number of frames:", len(frames))




# ===== Code Cell 13 =====
NUM_FRAMES = len(frames)
NUM_FRAMES

# ===== Code Cell 14 =====
frames[0].min(axis=0).min(axis=0), frames[0].max(axis=0).max(axis=0)

# ===== Markdown Cell 15 =====
# Override sampling frequency

# ===== Code Cell 16 =====
print(f"Detected Video Sampling rate: {fs}")

if MANUAL_FS:
    print(f"Overriding to: {MANUAL_FS}")
    fs = MANUAL_FS
    VIDEO_FS = fs
else:
    VIDEO_FS = fs

# ===== Markdown Cell 17 =====
# ## Get Temporal Filter

# ===== Code Cell 18 =====
import scipy.signal as signal


bandpass = signal.firwin(numtaps=NUM_FRAMES,
                         cutoff=(f_lo, f_hi),
                         fs=fs,
                         pass_zero=False)

# ===== Code Cell 19 =====
transfer_function = np.fft.fft(np.fft.ifftshift(bandpass))

# ===== Code Cell 20 =====
plt.plot(np.abs(transfer_function))
plt.title("Transfer Function");

# ===== Code Cell 21 =====
plt.plot(bandpass)
plt.title("Impulse Response");

# ===== Code Cell 22 =====
norm_freqs, response = signal.freqz(bandpass)
freqs = norm_freqs / np.pi * fs/ 2 

_, ax = plt.subplots(2, 1, figsize=(15, 7))
ax[0].plot(freqs, 20*np.log10(np.abs(response)));
ax[0].plot([f_lo, f_lo], [-100, -10], color='m')
ax[0].plot([f_hi, f_hi], [-100, -10], color='m')
ax[0].set_title("Frequency Response");
ax[0].set_ylabel("Amplitude");

ax[1].plot(freqs, np.angle(response));
ax[1].set_title("Phase Response");
ax[1].set_xlabel("Freqeuncy (Hz)");
ax[1].set_ylabel("Angle (radians)");

# ===== Markdown Cell 23 =====
# ### Gaussian Pyramid

# ===== Code Cell 24 =====
def gaussian_pyramid(image, level):
    """ Obtains single band of a Gaussian Pyramid Decomposition
        Inputs: 
            image - single channel input image
            num_levels - number of pyramid levels
        Outputs:
            pyramid - Pyramid decomposition tensor
        """ 
    rows, cols, colors = image.shape
    scale = 2**level
    pyramid = np.zeros((colors, rows//scale, cols//scale))

    for i in range(0, level):
        # image = cv2.pyrDown(image)

        image = cv2.pyrDown(image, dstsize=(cols//2, rows//2))
        rows, cols, _ = image.shape

        if i == (level - 1):
            for c in range(colors):
                pyramid[c, :, :] = image[:, :, c]

    return pyramid

# ===== Code Cell 25 =====
rows, cols, colors = frames[0].shape
scale = 2**LEVEL
pyramid_stack = np.zeros((NUM_FRAMES, colors, rows//scale, cols//scale))

# ===== Code Cell 26 =====
for i, frame in enumerate(frames):
    pyramid = gaussian_pyramid(frame, LEVEL)
    pyramid_stack[i, :, :, :] = pyramid

# ===== Code Cell 27 =====
plt.imshow(pyramid_stack[0, :, :, :].transpose(1, 0, 2).reshape((pyramid.shape[1], -1)), cmap='gray');

# ===== Markdown Cell 28 =====
# #### Apply Temporal Filtering

# ===== Code Cell 29 =====
pyr_stack_fft = np.fft.fft(pyramid_stack, axis=0).astype(np.complex64)
_filtered_pyramid = pyr_stack_fft * transfer_function[:, None, None, None].astype(np.complex64)
filtered_pyramid = np.fft.ifft(_filtered_pyramid, axis=0).real

# ===== Code Cell 30 =====
pyr_stack_fft.shape

# ===== Code Cell 31 =====
_, ax = plt.subplots(2, 1, figsize=(10, 5), sharey=True)

ax[0].plot(np.abs(pyr_stack_fft[2:-2, 0, 20, 12]))
ax[0].set_title("Unfiltered Signal at (20, 12)")

ax[1].plot(np.abs(_filtered_pyramid[2:-2, 0, 20, 12]))
ax[1].set_title("Filtered Signal at (20, 12)");

plt.tight_layout();

# ===== Code Cell 32 =====
_, ax = plt.subplots(1, 2)
ax[0].imshow(pyramid_stack[50, 0, :, :], cmap='gray')
ax[0].set_title("Unfiltered Luma Channel")
ax[1].imshow(filtered_pyramid[50, 0, :, :], cmap='gray')
ax[1].set_title("Filtered Luma Channel");

# ===== Markdown Cell 33 =====
# Display filtered results at single pixel

# ===== Code Cell 34 =====
plt.plot(pyramid_stack[:, 0, 12, 20] - pyramid_stack[:, 0, 12, 20].mean())
plt.plot(filtered_pyramid[:, 0, 12, 20]);

# ===== Markdown Cell 35 =====
# ## Apply Magnification and Reconstruct Video

# ===== Code Cell 36 =====
magnified_pyramid = filtered_pyramid * ALPHA

# ===== Code Cell 37 =====
magnified = []
magnified_only = []

for i in range(NUM_FRAMES):
    y_chan = frames[i][:, :, 0]
    i_chan = frames[i][:, :, 1] 
    q_chan = frames[i][:, :, 2] 
    
    fy_chan = cv2.resize(magnified_pyramid[i, 0, :, :], (cols, rows))
    fi_chan = cv2.resize(magnified_pyramid[i, 1, :, :], (cols, rows))
    fq_chan = cv2.resize(magnified_pyramid[i, 2, :, :], (cols, rows))

    # apply magnification
    mag = np.dstack((
        y_chan + fy_chan,
        i_chan + fi_chan,
        q_chan + fq_chan,
    ))
    
    # normalize and convert to RGB
    mag = inv_colorspace(mag)

    # store magnified frames
    magnified.append(mag)

    # store magified only for reference
    magnified_only.append(np.dstack((fy_chan, fi_chan, fq_chan)))

# ===== Markdown Cell 38 =====
# Check detected heart rates

# ===== Code Cell 39 =====
og_reds = []
og_blues = []
og_greens = []

reds = []
blues = []
greens = []
for i in range(NUM_FRAMES):
    # convert YIQ to RGB
    frame = inv_colorspace(frames[i])
    og_reds.append(frame[0, :, :].sum())
    og_blues.append(frame[1, :, :].sum())
    og_greens.append(frame[2, :, :].sum())

    reds.append(magnified[i][0, :, :].sum())
    blues.append(magnified[i][1, :, :].sum())
    greens.append(magnified[i][2, :, :].sum())

# ===== Code Cell 40 =====
times = np.arange(0, NUM_FRAMES)/fs

# ===== Code Cell 41 =====
fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
ax[0].plot(times, og_reds, color='red')
ax[0].plot(times, og_blues, color='blue')
ax[0].plot(times, og_greens, color='green')
ax[0].set_title("Original", size=18)
ax[0].set_xlabel("Time", size=16)
ax[0].set_ylabel("Intensity", size=16)

ax[1].plot(times, reds, color='red')
ax[1].plot(times, blues, color='blue')
ax[1].plot(times, greens, color='green')
ax[1].set_title("Filtered", size=18)
ax[1].set_xlabel("Time", size=16);

# ===== Code Cell 42 =====
freqs = np.fft.rfftfreq(NUM_FRAMES) * fs
rates = np.abs(np.fft.rfft(reds))/NUM_FRAMES

# ===== Code Cell 43 =====
plt.plot(freqs[1:], rates[1:]);
plt.title("DFT of Red channel Intensities")
plt.xlabel("Freuqency")
plt.ylabel("Amplitude");

# ===== Markdown Cell 44 =====
# find peak

# ===== Code Cell 45 =====
peak_idx, _ = signal.find_peaks(rates, height=1000)

# ===== Code Cell 46 =====
freqs[peak_idx], rates[peak_idx]

# ===== Code Cell 47 =====
bpm = freqs[peak_idx].squeeze(0) * 60
bpm

# ===== Markdown Cell 48 =====
# ## Make a video

# ===== Code Cell 49 =====
stacked_frames = []
middle = np.zeros((rows, 3, 3)).astype(np.uint8)

for vid_idx in range(NUM_FRAMES):
    og_frame = cv2.normalize(yiq2rgb(frames[vid_idx]), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
    frame = np.hstack((cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR), 
                       middle, 
                       cv2.cvtColor(magnified[vid_idx], cv2.COLOR_RGB2BGR)))
    stacked_frames.append(frame)

# ===== Code Cell 50 =====
plt.imshow(cv2.cvtColor(stacked_frames[10], cv2.COLOR_BGR2RGB));

# ===== Code Cell 51 =====
# get width and height for video frames
_h, _w, _ = stacked_frames[-1].shape

# save to mp4
out = cv2.VideoWriter(f"stacked_{int(ALPHA)}x.mp4",
                      cv2.VideoWriter_fourcc(*'MP4V'), 
                      int(fs), 
                      (_w, _h))
 
for frame in stacked_frames:
    out.write(frame)

out.release()
del out

# ===== Markdown Cell 52 =====
# Create a video of the magnified only frames

# ===== Code Cell 53 =====
# get width and height for video frames
_h, _w, _ = magnified_only[-1].shape

# save to mp4
out = cv2.VideoWriter(f"stacked_{int(ALPHA)}x_AMP.mp4",
                      cv2.VideoWriter_fourcc(*'MP4V'), 
                      int(fs), 
                      (_w, _h))

sums = []
for frame in magnified_only:
    sums.append(frame.sum(axis=1).sum(axis=0))
    
    frame = cv2.cvtColor(
        cv2.normalize(frame*20, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1),
        cv2.COLOR_RGB2BGR)
    out.write(frame)

out.release()
del out

# ===== Markdown Cell 54 =====
# Create GIF

# ===== Code Cell 55 =====
h, w, _ = stacked_frames[0].shape

# ===== Code Cell 56 =====
h2 = np.round(h/2.5).astype(int)
w2 = np.round(w/2.5).astype(int)

# ===== Code Cell 57 =====
from PIL import Image 


# accumulate PIL image objects
pil_images = []
for img in stacked_frames:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(w2, h2))
    pil_images.append(Image.fromarray(img))

# create GIF
pil_images[0].save(f"stacked_{int(ALPHA)}x.gif", 
                    format="GIF", 
                    append_images=pil_images, 
                    save_all=True, 
                    duration=50, # duration that each frame is displayed
                    loop=0)

# ===== Markdown Cell 58 =====
# ## Visualize Amplification

# ===== Code Cell 59 =====
stacked = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in stacked_frames])

# ===== Code Cell 60 =====
frame.shape

# ===== Code Cell 61 =====
frame = stacked[0, :, :, :].copy()[:, :og_w, :]

forehead_idx = 235
lcheek_idx = 100
rcheek_idx = 355

cv2.line(frame, (forehead_idx, 10), (forehead_idx, 200), (0,255,0), 5)
cv2.line(frame, (lcheek_idx, 285), (lcheek_idx, 425), (0,255,0), 5)
cv2.line(frame, (rcheek_idx, 285), (rcheek_idx, 425), (0,255,0), 5)

plt.imshow(frame);
plt.title("Locations");

# ===== Code Cell 62 =====
idx1 = 220 
idx2 = idx1 + og_w + 3

# ===== Code Cell 63 =====
fig, ax = plt.subplots(1, 2, figsize=(7,5))
fig.suptitle("Middle", size=22)
ax[0].imshow(stacked[:, :, idx1, :].transpose(1, 0, 2))
ax[0].set_title("Original Image")
ax[1].imshow(stacked[:, :, idx2, :].transpose(1, 0, 2))
ax[1].set_title("Color Magnified");

plt.tight_layout();

# ===== Code Cell 64 =====
fig, ax = plt.subplots(1, 2, figsize=(7,3))
fig.suptitle("Forehead", size=22)
ax[0].imshow(stacked[:, 10:200, forehead_idx, :].transpose(1, 0, 2))
ax[0].set_title("Original Image")
ax[1].imshow(stacked[:, 10:200, forehead_idx + og_w + 3, :].transpose(1, 0, 2))
ax[1].set_title("Color Magnified");

plt.tight_layout();

# ===== Code Cell 65 =====
fig, ax = plt.subplots(2, 2, figsize=(10, 5))
fig.suptitle("Cheeks", size=22)
ax[0, 0].imshow(stacked[:, 285:425, lcheek_idx, :].transpose(1, 0, 2))
ax[0, 0].set_title("Original Image (Left Cheek)")
ax[0, 1].imshow(stacked[:, 285:425, lcheek_idx + og_w + 3, :].transpose(1, 0, 2))
ax[0, 1].set_title("Color Magnified");
ax[1, 0].imshow(stacked[:, 285:425, rcheek_idx, :].transpose(1, 0, 2))
ax[1, 0].set_title("Original Image (Right Cheek)")
ax[1, 1].imshow(stacked[:, 285:425, rcheek_idx + og_w + 3, :].transpose(1, 0, 2))
ax[1, 1].set_title("Color Magnified");

plt.tight_layout();

# ===== Code Cell 66 =====

