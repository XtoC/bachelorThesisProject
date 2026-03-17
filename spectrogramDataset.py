import os
import numpy as np
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    def __init__(self, features_path):
        self.alarm_tpyes = ["Smoke_alarm", "Fire_alarm_bell", "Fire_alarm_electronic", "Air_siren",
                            "No_alarm_inside", "No_alarm_outside"]
        self.label_to_idx = {c: i for i,c in enumerate(self.alarm_tpyes)}

        self.data = []
        for dir in os.listdir(features_path):
            sub_dir = os.path.join(features_path, dir)
            if os.path.isdir(sub_dir):
                for filename in os.listdir(sub_dir):
                    if filename == ".DS_Store":
                        continue
                    file_path = os.path.join(sub_dir, filename)
                    x = np.load(file_path)
                    y = self.label_to_idx[dir]
                    self.data.append((x,y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]