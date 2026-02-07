import librosa
import soundfile as sf
import numpy as np
import os

def normalize_audio_to_wav(input_path, output_path, target_sr=44100):
    # Load audio with librosa
    audio, sr = librosa.load(input_path, sr=None)  # Load original sample rate

    # Resample to target sample rate
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Save into wav files
    sf.write(output_path, audio, target_sr, subtype='PCM_16')


if __name__ == "__main__":

    input_folder = "Alarms"
    output_folder = "audio/normalized_audio"
    for dir in os.listdir(input_folder):
        sub_dir = os.path.join(input_folder, dir)
        if os.path.isdir(sub_dir):
            for filename in os.listdir(sub_dir):
                if filename == ".DS_Store":
                    continue
                input_path = os.path.join(sub_dir, filename)
                output_path = os.path.join(output_folder, dir)
                name_no_ext = os.path.splitext(filename)[0]
                output_path = os.path.join(output_path, name_no_ext + ".wav")
                # print("***input path***")
                # print(input_path)
                # print("***output path***")
                # print(output_path)
                # print()
                normalize_audio_to_wav(input_path, output_path)

