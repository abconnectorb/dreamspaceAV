import numpy as np
import librosa
import matplotlib.pyplot as plt

# Configuration #
MP3_FILE = "Farsighted.mp3"  # Change this to your MP3 file path
#################

OUTPUT_FILE = "frequency_bands.npy"  # Numpy file to save the data
SAMPLE_RATE = 22050  # Audio sample rate
FPS = 60  # Match your visualization FPS
NUM_BANDS = 20  # Number of frequency bands across the spectrum

def extract_frequency_bands(mp3_path, output_path):
    """
    Extract multiple frequency bands from an MP3 file.
    Each band will correspond to a position along the x-axis.
    """
    print(f"Loading {mp3_path}...")
    
    # Load audio file
    y, sr = librosa.load(mp3_path, sr=SAMPLE_RATE)
    duration = len(y) / sr
    
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample rate: {sr} Hz")
    
    # Calculate hop length to match FPS
    hop_length = int(sr / FPS)
    
    # Compute Short-Time Fourier Transform (STFT)
    print("Computing STFT...")
    stft = librosa.stft(y,n_fft=4096, hop_length=hop_length, window='hann')
    
    # Get magnitude spectrogram
    magnitude = np.abs(stft)
    
    # Convert to decibels
    db_spec = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # Define frequency bands logarithmically (human hearing is logarithmic)
    # From 20 Hz (sub-bass) to 20000 Hz (high treble)
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=4096)
    min_freq = 20
    max_freq = 16000  # Focus on audible range
    
    # Create logarithmic frequency bands
    band_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), NUM_BANDS + 1)
    
    print(f"\nFrequency bands:")
    for i in range(NUM_BANDS):
        print(f"Band {i}: {band_edges[i]:.1f} - {band_edges[i+1]:.1f} Hz")
    
    # Extract amplitude for each band over time
    band_amplitudes = []
    
    for i in range(NUM_BANDS):
        low_freq = band_edges[i]
        high_freq = band_edges[i + 1]
        
        # Find frequency bin indices for this band
        band_mask = (freq_bins >= low_freq) & (freq_bins < high_freq)
        
        if np.any(band_mask):
            # Average amplitude in this frequency band over time
            band_energy = np.mean(db_spec[band_mask, :], axis=0)
        else:
            band_energy = np.zeros(db_spec.shape[1])
        
        # Normalize to 0-1 range
        if band_energy.max() > band_energy.min():
            band_energy = (band_energy - band_energy.min()) / (band_energy.max() - band_energy.min())
        else:
            band_energy = np.zeros_like(band_energy)
        
        band_amplitudes.append(band_energy)
    
    band_amplitudes = np.array(band_amplitudes)  # Shape: (NUM_BANDS, num_frames)
    
    # Smooth the data slightly to avoid jitter
    from scipy.ndimage import gaussian_filter1d
    for i in range(NUM_BANDS):
        band_amplitudes[i] = gaussian_filter1d(band_amplitudes[i], sigma=1.5)
    
    # Save data
    np.save(output_path, band_amplitudes)
    
    print(f"\nSaved frequency band data to {output_path}")
    print(f"Shape: {band_amplitudes.shape} (bands x frames)")
    print(f"Number of frames: {band_amplitudes.shape[1]}")
    print(f"Expected duration at {FPS} FPS: {band_amplitudes.shape[1] / FPS:.2f} seconds")
    
    # Plot the frequency bands over time
    plt.figure(figsize=(14, 8))
    
    # Plot band amplitudes as heatmap
    plt.subplot(2, 1, 1)
    time_axis = np.arange(band_amplitudes.shape[1]) / FPS
    plt.imshow(band_amplitudes, aspect='auto', origin='lower', cmap='magma', 
               extent=[0, time_axis[-1], 0, NUM_BANDS])
    plt.colorbar(label='Normalized Amplitude')
    plt.title("Frequency Bands Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency Band (Low to High)")
    
    # Plot spectrogram
    plt.subplot(2, 1, 2)
    librosa.display.specshow(db_spec, sr=sr, hop_length=hop_length, 
                             x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Full Spectrogram')
    plt.ylim(0, max_freq)
    
    plt.tight_layout()
    plt.savefig('frequency_bands_analysis.png', dpi=150)
    print("Saved visualization to frequency_bands_analysis.png")
    plt.show()
    
    return band_amplitudes

if __name__ == "__main__":
    # Extract frequency data
    freq_data = extract_frequency_bands(MP3_FILE, OUTPUT_FILE)
    
    print("\n" + "="*50)
    print("Frequency band data extracted successfully!")
    print("="*50)