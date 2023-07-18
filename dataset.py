from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class JY_Dataset(Dataset):
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.total_size = len(self.dataset)
        self.queue = [*range(self.total_size)]
        self.config = config
        
    def __getitem__(self, index):
        p = self.queue[index]
        waveform  = self.dataset[p]["waveform"]
        while len(waveform) < self.config.clip_samples: # 10초보다 짧으면 concat
            waveform = np.concatenate((waveform, waveform))
        waveform = waveform[:self.config.clip_samples] # 10초로 자르기
        
        target = np.zeros(self.config.classes_num).astype(np.float32)
        target[int(self.dataset[p]["target"])] = 1.
        data_dict = {
            "audio_name": self.dataset[p]["name"],
            "waveform": waveform,
            "real_len": len(waveform),
            "target": target
        }
        return data_dict

    def __len__(self):
        return self.total_size