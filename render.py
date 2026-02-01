import cv2
import numpy as np
from scipy.interpolate import interp1d

WIDTH, HEIGHT = 1280, 720
FPS = 60

#CONFIG ZONE
#SPEED OF 1 is 123 (at verticle scale of 0.5) BPM! or close to it
bpm = 78

colorSpeed = 0.5

#howm much the loudness affects the waves size. 0 is no influence, 1 is quite a bit
loudnessInfluence = 0.8

#vertical scale allows more waves on screen at once. Ingame version is about 0.8
verticalScale = 0.6
#TS TAKES FOREVER (5 min per 1 min of Video at 60fps on my PC)
#END CONFIG ZONE

speed = bpm / 123

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('DreamspaceAV.mp4', fourcc, FPS, (WIDTH, HEIGHT))

# Colors in BGR
brightMod = 10

BACKGROUND = np.array([130, 142, 56], dtype=np.uint8)
WAVE_COLOR = np.array([brightMod, brightMod, brightMod], dtype=np.uint8)

# DS resolution
DS_WIDTH, DS_HEIGHT = 256, 192

# Load frequency band data
FREQUENCY_BANDS = np.load("frequency_bands.npy")  # Shape: (NUM_BANDS, num_frames)
NUM_BANDS = FREQUENCY_BANDS.shape[0]
print(f"Loaded {FREQUENCY_BANDS.shape[1]} frames of frequency data with {NUM_BANDS} bands")



midSpeed = speed * -1

#veritcal scale preserves bpm :)

frequency = 6.0 / verticalScale

amplitude = int((HEIGHT // 12) * verticalScale)
midAmplitude = int((HEIGHT // 14) * verticalScale)

frame_count = 0
x = np.arange(WIDTH)

# --- Thickness setting ---
THICKNESS = int(275 * verticalScale)
MIDTHICKNESS = int(200 * verticalScale)

# --- AMPLITUDE MULTIPLIER SETTINGS ---
AMPLITUDE_SCALE = 2.0  # How much the audio affects amplitude (1.0 = no effect, 2.0 = doubles range)

# PRECOMPUTE ALL BACKGROUND COLORS
NUM_COLORS = 360
BACKGROUND_COLORS = np.zeros((NUM_COLORS, 3), dtype=np.uint8)

# this is for lighter purples and blues
for i in range(NUM_COLORS):
    hue = (i * 180) // NUM_COLORS
    
    if 90 <= hue <= 180:
        if hue <= 130:
            t = (hue - 90) / 45
        else:
            t = (180 - hue) / 45
        
        ease = (1 - np.cos(t * np.pi)) / 2
        saturation = int(255 - ease * 105)
        value = int(142 + ease * 38)
    else:
        saturation = 255
        value = 142
    
    hsv_color = np.array([[[hue, saturation, value]]], dtype=np.uint8)
    BACKGROUND_COLORS[i] = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]

while True:
    # Look up precomputed background color
    color_index = int((frame_count * colorSpeed) % NUM_COLORS)
    BACKGROUND = BACKGROUND_COLORS[color_index]
    
    # Create background frame
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = BACKGROUND

    t = frame_count / FPS
    
    # Get current frequency bands (loop if video is longer than audio)
    audio_frame = frame_count % FREQUENCY_BANDS.shape[1]
    current_bands = FREQUENCY_BANDS[:, audio_frame]  # Shape: (NUM_BANDS,)
    
    # Interpolate frequency bands across the WIDTH of the screen
    # Bass (low frequencies) on the left, treble (high frequencies) on the right
    band_positions = np.linspace(0, WIDTH - 1, NUM_BANDS)
    interpolator = interp1d(band_positions, current_bands, kind='linear', fill_value='extrapolate')
    amplitude_multiplier = 1 + (AMPLITUDE_SCALE - 1) * interpolator(x)
    
    # Clamp to reasonable values
    amplitude_multiplier = loudnessInfluence * np.clip(amplitude_multiplier, 0.5, 3.0)

    # Vectorized y center positions with audio-reactive amplitude multiplier
    y_center = (HEIGHT // 2 + ((np.sin(2*(x / WIDTH * frequency - speed * t)) + 5) * amplitude * amplitude_multiplier * np.sin(
        2 * np.pi * (x / WIDTH * frequency - speed * t)
    ))/5).astype(np.int32)

    # Create accumulator for wave brightness
    wave_layer = np.zeros((HEIGHT, WIDTH), dtype=np.int16)

    # OUTER WAVE
    layers = [(1.0, 1), (0.7, 2), (0.4, 3), (0.2, 4)]

    for mult, brightness in layers:
        thick = int(THICKNESS * mult)
        
        for i in range(WIDTH):
            # Apply position-specific audio amplitude to thickness
            adjusted_thick = int(thick * amplitude_multiplier[i])
            y_start = max(0, (y_center[i] - adjusted_thick)) 
            y_end = min(HEIGHT, (y_center[i] + adjusted_thick + 1)) 
            wave_layer[y_start:y_end, i] += brightMod * brightness

    # INNER WAVE
    mid_y_center = (HEIGHT // 2 + ((np.sin(2*(x / WIDTH * frequency - midSpeed * t)) + 5) * midAmplitude * amplitude_multiplier * np.sin(
        2 * np.pi * (x / WIDTH * frequency - midSpeed * t)
    ))/5).astype(np.int32)

    mid_layers = [(1.0, 1), (0.7, 2), (0.4, 3), (0.2, 4)]

    for mult, brightness in mid_layers:
        thick = int(MIDTHICKNESS * mult)
        
        for i in range(WIDTH):
            # Apply position-specific audio amplitude to thickness
            adjusted_thick = int(thick * amplitude_multiplier[i])
            y_start = max(0, mid_y_center[i] - adjusted_thick)
            y_end = min(HEIGHT, mid_y_center[i] + adjusted_thick + 1)
            wave_layer[y_start:y_end, i] += brightMod * brightness

    # Apply wave layer to frame
    wave_layer = np.minimum(wave_layer, 255).astype(np.uint8)
    frame[:, :] = np.minimum(frame.astype(np.int16) + wave_layer[:, :, np.newaxis], 255).astype(np.uint8)

    # DS RESOLUTION EFFECT
    # Downscale to DS resolution
    #frame = cv2.resize(frame, (DS_WIDTH, DS_HEIGHT), interpolation=cv2.INTER_AREA)

    #UPSCALE DS RESOLUTION EFFECT back to original size with nearest neighbor for pixelated look
    #frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)


    if frame_count % 300 == 0:
        print(f"Rendered {frame_count}/{FREQUENCY_BANDS.shape[1]} frames")
    out.write(frame)

    #live preview (optional)
    #cv2.imshow("Audio-Reactive Sine Wave", frame)
    #if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
    #    break

    frame_count += 1
    if frame_count >= FREQUENCY_BANDS.shape[1]:  # stop after all frames
        break

cv2.destroyAllWindows()

out.release()
print("Video saved as DreamspaceAV.mp4")