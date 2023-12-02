### DATA API - Time series examples ###

from pykrx import stock

start = "20000101"
end = "20231001"

df = stock.get_market_ohlcv(start, end, "005930")  # Samsung

### SMA 추가 ###
df['MA-5'] = df['종가'].rolling(window=5).mean()
df['MA-15'] = df['종가'].rolling(window=15).mean()

### EMA 추가 ###
span_5 = 5   # 평활계수 a = 2 / (span + 1)
df['EMA-5'] = df['종가'].ewm(span=span_5, adjust=False).mean()
span_15 = 15   
df['EMA-15'] = df['종가'].ewm(span=span_15, adjust=False).mean()

### RSI 계산 ###
def calculate_rsi(data, window=14):
    diff = data['종가'].diff(1)
    gain = diff.where(diff > 0, 0)
    loss = -diff.where(diff < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
df['RSI'] = calculate_rsi(df)

"""
### 스토캐스틱 오실레이터 계산 ###
def calculate_stochastic_oscillator(data, period_k, period_d):
    low_min = data['저가'].rolling(window=period_k).min()
    high_max = data['고가'].rolling(window=period_k).max()
    data['%K'] = (data['종가'] - low_min) / (high_max - low_min) * 100
    data['%D'] = data['%K'].rolling(window=period_d).mean()
period_k = 14
period_d = 3
#calculate_stochastic_oscillator(df, period_k, period_d)

### 볼린저 밴드 계산 ###
def calculate_bollinger_bands(data, window, num_std_dev):
    rolling_mean = data['종가'].rolling(window=window).mean()
    rolling_std = data['종가'].rolling(window=window).std()
    data['Upper_BB'] = rolling_mean + (rolling_std * num_std_dev)
    data['Lower_BB'] = rolling_mean - (rolling_std * num_std_dev)
window = 20
num_std_dev = 2
#calculate_bollinger_bands(df, window, num_std_dev)
"""

### 값이 0이 포함된 데이터 행 제거 / 결측치 제거 ###
df = df[(df != 0).all(axis=1)]
df = df.dropna()

### 등락률, 거래량 제거 ###
df.drop('시가', axis=1, inplace=True)
df.drop('등락률', axis=1, inplace=True)
df.drop('거래량', axis=1, inplace=True)

print(f"df1_daily.shape: {df.shape}")
print(df)

df.to_csv(f'./TCN/data/df.csv')
