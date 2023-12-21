import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable

def perform_audio_analysis(y, sr):
    # Calculate the spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Perform octave analysis
    octaves = librosa.core.hz_to_octs(np.linspace(librosa.note_to_hz('C1'), librosa.note_to_hz('C8'), 12))
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Convert centroid frequencies to octaves
    centroid_octaves = librosa.hz_to_octs(centroid[0])

    # Get octave band names
    octave_band_names = [f'C{i}' for i in range(1, len(octaves) + 1)]

    # Create a figure and axes
    fig, axs = plt.subplots(5, 1, figsize=(12, 18))

    # Plot the spectrogram
    im = axs[0].imshow(D, aspect='auto', cmap='viridis', origin='lower', extent=[0, D.shape[1], 0, D.shape[0]])
    cbar = plt.colorbar(im, ax=axs[0], format='%+2.0f dB')
    cbar.set_label('Intensity (dB)')
    axs[0].set_title('Spectrogram')

    # Plot the octave analysis
    axs[1].plot(librosa.times_like(centroid_octaves), centroid_octaves, label='Octave Analysis')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Octave Centroid')
    axs[1].set_title('Octave Analysis')
    axs[1].legend()

    # Calculate power in each octave band with a finite range
    power_per_band, _ = np.histogram(centroid_octaves, bins=len(octaves) - 1, range=(0, 8))

    # Ensure that the length of octave_band_names matches the length of power_per_band
    octave_band_names = octave_band_names[:len(power_per_band)]

    # Plot the histogram of power in octave bands
    axs[2].bar(octave_band_names, power_per_band, align='edge')
    axs[2].set_xlabel('Octave Bands')
    axs[2].set_ylabel('Power')
    axs[2].set_title('Power Distribution in Octave Bands')

    # Extract the melody
    melody, _ = librosa.core.piptrack(y=y, sr=sr)

    # Plot the melody
    im_melody = axs[3].imshow(librosa.amplitude_to_db(melody, ref=np.max), aspect='auto', cmap='viridis',
                              origin='lower', extent=[0, melody.shape[1], 0, melody.shape[0]])
    cbar_melody = plt.colorbar(im_melody, ax=axs[3], format='%+2.0f dB')
    cbar_melody.set_label('Intensity (dB)')
    axs[3].set_title('Melody Analysis')

    # Compute chromagram
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Plot the chromagram
    im_chroma = axs[4].imshow(chroma, aspect='auto', cmap='viridis', origin='lower',
                              extent=[0, chroma.shape[1], 0, chroma.shape[0]])
    cbar_chroma = plt.colorbar(im_chroma, ax=axs[4])
    cbar_chroma.set_label('Intensity')
    axs[4].set_title('Harmonic Analysis (Chromagram)')

    # Adjust layout
    plt.tight_layout()

    # Display the figure using Streamlit
    st.pyplot(fig)

def main():
    st.title("Audio Analysis App by DEI Multimedia Team")

    uploaded_file = st.file_uploader("Choose an MP3 file", type="mp3")

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/mp3', start_time=0)

        if st.button("Generate Plots"):
            # Hide warning
            st.set_option('deprecation.showPyplotGlobalUse', False)
            
            y, sr = librosa.load(uploaded_file, sr=None)
            st.subheader("Audio Analysis Results")
            perform_audio_analysis(y, sr)

if __name__ == "__main__":
    main()
