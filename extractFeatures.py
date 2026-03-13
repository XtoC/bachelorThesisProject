import librosa
import os
import numpy as np

Alarm_types = ["Smoke_alarm", "Fire_alarm_bell", "Fire_alarm_electronic", "Air_siren"]

def log_mel_spectrogram(
    in_path: str,
    out_path: str,
    sr_target: int = 24000,
    n_fft: int = 2048,
    hop_length: int = 1200,
    win_length: int | None = None,
    n_mels: int = 64,
    fmin: float = 20.0,
    fmax: float | None = None,  # default None -> sr_target/2
    power: float = 2.0,         # power=2 for power spectrogram
    ref: float = 1.0,           # reference for amplitude conversion
    eps: float = 1e-10
):
    # 1) Load
    y, sr = librosa.load(in_path, sr=sr_target, mono=True)

    # 2) Mel spectrogram (power)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr_target,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power
    )

    # 3) Log compression (dB)
    S_db = librosa.power_to_db(S, ref=ref, top_db=None)  # dB scale

    np.save(out_path, S_db.astype(np.float32))

if __name__ == "__main__":
    soundscapes_folder = "audio/soundscapes"
    features_folder = "audio/features_audio"
    for dir in os.listdir(soundscapes_folder):
        sub_dir_soundscapes = os.path.join(soundscapes_folder, dir)
        sub_dir_features = os.path.join(features_folder, dir)
        if os.path.isdir(sub_dir_soundscapes):
            for dir in os.listdir(sub_dir_soundscapes):
                alarm_dir_soundscapes = os.path.join(sub_dir_soundscapes, dir)
                alarm_dir_features = os.path.join(sub_dir_features, dir)
                if os.path.isdir(alarm_dir_soundscapes):
                    for filename in os.listdir(alarm_dir_soundscapes):
                        if not filename.lower().endswith(".wav"):
                            continue
                        file_inpath = os.path.join(alarm_dir_soundscapes, filename)
                        file_outpath = os.path.join(alarm_dir_features, filename.replace(".wav", ".npy"))
                        log_mel_spectrogram(file_inpath, file_outpath)