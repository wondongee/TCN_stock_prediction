import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

### 데이터 로더 ###
class Loader32(Dataset):
    def __init__(self, data, length, window):
        assert len(data) >= length + window
        self.data = data
        self.length = length
        self.window = window
    def __getitem__(self, idx):
        if idx + self.length + self.window < len(self.data):
            x = torch.tensor(self.data[idx:idx+self.length]).reshape(-1, self.length).to(torch.float32)
            start_idx = idx+self.length
            end_idx = idx+self.length+self.window
            # 누적 로그리턴 예측일 경우
            # {MA-7 : 3 or EMA-7 : 5}
            y = torch.tensor( np.sum(self.data[start_idx:end_idx, 3]) ).to(torch.float32)
            # 가격 데이터 예측일 경우
            # y = torch.tensor( np.sum(self.data[idx+ self.length + self.window, 3]) ).to(torch.float32)
            return x, y
        else:
            return None
    def __len__(self):
        return max(len(self.data)-self.length-self.window, 0)
