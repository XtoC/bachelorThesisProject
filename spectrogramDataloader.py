from spectrogramDataset import SpectrogramDataset
from torch.utils.data import DataLoader

train_ds = SpectrogramDataset("audio/features_audio/train_audio")
val_ds = SpectrogramDataset("audio/features_audio/validation_audio")
test_ds = SpectrogramDataset("audio/features_audio/test_audio")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
