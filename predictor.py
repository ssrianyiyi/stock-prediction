import pandas as PD
import numpy as np
# %matplotlib inline
import matplotlib. pyplot as plt
import matplotlib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib. dates as mandates
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM
from keras.utils import plot_model
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

whole_data = pd.read_csv("prices-split-adjusted.csv")
df = pd.read_csv("prices-split-adjusted.csv", header=0, index_col='date', parse_dates=True)
target_symbol = 'AAPL'
filtered_df = df[df['symbol'] == target_symbol]

print(filtered_df.head())

print("Filtered Dataframe Shape:", filtered_df.shape)
print("Null Value Present in Filtered DataFrame:", filtered_df.isnull().values.any())

filtered_df['close'].plot()
plt.title(f'{target_symbol} Stock Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

output_var = pd.DataFrame(filtered_df['close'])
features = ['open', 'close', 'high', 'low', 'volume']

scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(filtered_df[features])
feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=filtered_df.index)
feature_transform.head()

test_index = round(len(feature_transform) * 0.8)
X_train, X_test = feature_transform[:test_index], feature_transform[test_index:]
y_train, y_test = output_var[:test_index].values.ravel(), output_var[test_index:].values.ravel()
print(y_train)

naive_predictions = np.roll(y_test, -1)
naive_predictions[-1] = y_test[-1]
naive_mse = mean_squared_error(y_test, naive_predictions)
print("Naive Model MSE:", naive_mse)

trainX = np.array(X_train)
testX = np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

lstm = Sequential()
lstm.add(LSTM(108, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
plot_model(lstm, show_shapes=True, show_layer_names=True)

lstm.fit(X_train, y_train, epochs=150, batch_size=1, verbose=1)

y_pred = lstm.predict(X_test)

plt.plot(y_test, label='True Value')
plt.plot(y_pred, label='LSTM Value')
plt.plot(naive_predictions, label='Naive Value')
plt.title("Prediction Comparison: LSTM vs Naive")
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend()
plt.show()

