import librosa
import soundfile as sf
import numpy as np


def normalize_audio_to_wav(input_path, output_path, target_sr=44100):
    # Load audio with librosa
    audio, sr = librosa.load(input_path, sr=None)  # Load original sample rate

    # Resample to target sample rate
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Normalize the amplitude
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Save into wav files
    sf.write(output_path, audio, target_sr, subtype='PCM_16')


if __name__ == "__main__":
    input_file = "input_audio.mp3"
    output_file = "normalized_output.wav"
    normalize_audio_to_wav(input_file, output_file)

