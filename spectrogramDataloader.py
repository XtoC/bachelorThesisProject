from spectrogramDataset import SpectrogramDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import librosa
import numpy as np

if __name__ == "__main__":
    train_ds = SpectrogramDataset("audio/features_audio/train_audio")
    val_ds = SpectrogramDataset("audio/features_audio/validation_audio")
    test_ds = SpectrogramDataset("audio/features_audio/test_audio")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    x, y = next(iter(train_ds))
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    img = librosa.display.specshow(x, y_axis='linear', x_axis='time',
                                   sr=24000, ax=ax[0])

    lm = np.log(x + np.finfo(x.dtype).eps)
    librosa.display.specshow(lm, y_axis='log', sr=24000, hop_length=1200,
                             x_axis='time', ax=ax[1])
    ax[1].set(title='Log-frequency power spectrogram')
    ax[1].label_outer()
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    print(x.shape)
    print(y)

    x, y = next(iter(train_loader))
    print(x.shape)
    print(y.shape)

    plt.show()


