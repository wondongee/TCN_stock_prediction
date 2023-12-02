import os, sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from data_loader import *
import model_tcn
from model_tcn import *

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)

### 데이터 읽기 ###
file_name = './TCN/data/df1.csv'
data = pd.read_csv(file_name)
data['날짜'] = pd.to_datetime(data['날짜'])
data.set_index('날짜', inplace=True)


### 데이터 전처리 ###c
seq_len = 127
window = 20
price = data.to_numpy()
idx = data.index
print(price.shape)

### train / test 분리 ###
train_size = int(len(price) * 0.7)
train_X = price[ : train_size]
test_X = price[train_size : ]
train_idx = idx[seq_len + window : train_size]
test_idx = idx[train_size + seq_len + window:]
print("train length : ", len(train_X), "test length : ", len(test_X))

### 데이터 스케일링 ###
mean = np.mean(train_X, axis=0)
std = np.std(train_X, axis=0)
train_scaled = (train_X - mean) / std
test_scaled = (test_X - mean)/ std

### 하이퍼파라미터 ###
num_epochs = 200
batch_size = 256
clip= 1.0
lr = 0.0001
receptive_field_size = 127

### Load the data ###
dataset_train = Loader32(train_scaled, seq_len, window)
dataset_test = Loader32(test_scaled, seq_len, window)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

### 모델 학습 ###
def train(dataloader, model, loss_fn, optimizer, num_epochs):
    t = tqdm(range(num_epochs))
    for epoch in t:
        total_loss = 0
        for idx, (data, target) in enumerate(dataloader, 0):
            if data == None or target == None:
                continue
            optimizer.zero_grad()
            pred = model(data)
            loss = loss_fn(pred.reshape(-1), target)
            loss.backward()
            optimizer.step()
            for dp in model.parameters():
                dp.data.clamp_(-clip, clip)
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Total Loss: {total_loss}")

### TCN 모델 초기화 및 정의 ###
input_channels=5
output_size=1
kernel_size=2
dropout=0.05
model = TCN(input_channels, output_size, kernel_size, dropout)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
train(dataloader_train, model, loss_fn, optimizer, num_epochs)

### 모델 테스트 ###
pred_y = []     # 예측값
target_y = []   # 실제값
with torch.no_grad():
    model.eval()
    for idx, (data, target) in enumerate(dataloader_test, 0):
        if data == None or target == None:
            continue
        pred = model(data)
        pred_y.append(pred)
        target_y.append(target)

pred_y = torch.cat(pred_y, dim=0).numpy()
target_y = torch.cat(target_y, dim=0).numpy()

### inverse 스케일링
pred_y = (pred_y.reshape(-1) * std[3]) + mean[3]
target_y = (target_y * std[3]) + mean[3]

# 로그 수익률을 가격 데이터로 변환
"""
price_pred_y = []
price_target_y = []
for log_return in pred_y:
    price_pred_y.append(log_return)
for log_return in target_y:
    price_target_y.append(log_return)

price_pred_y = np.array(price_pred_y).flatten()
price_target_y = np.array(price_target_y).flatten()
print(price_pred_y.shape)
print(price_target_y.shape)
"""

df1 = pd.DataFrame(pred_y)
df2 = pd.DataFrame(target_y)
df_concat = pd.concat([df1, df2], axis=1)
df_concat.to_excel('output.xlsx', index=False)  # 'data.xlsx'는 저장할 파일명
print(df_concat)
"""
plt.figure(figsize=(15,6))
plt.plot(price_target_y, label='real')  # 첫 번째 시계열 데이터 플롯
plt.plot(price_pred_y, label='pred')  # 두 번째 시계열 데이터 플롯
plt.legend()
plt.grid(True)
plt.show()
"""