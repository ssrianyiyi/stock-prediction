# Importing the Libraries
import pandas as PD
import numpy as np
# %matplotlib inline
import matplotlib. pyplot as plt
import matplotlib
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




# Get the Dataset
apple_data = PD.read_csv("prices-split-adjusted.csv")
# column_names = ['date', 'symbol', 'open', 'close', 'low', 'high', 'volume']
df = PD.read_csv("prices-split-adjusted.csv", header=0, index_col='date', parse_dates=True)
print(df.head())

## todo: make df store only the data rows which have symbol = AAPL


# Print the shape of Dataframe and Check for Null Values
print("Dataframe Shape:", df.shape)
print("Null Value Present:", df.isnull().values.any())


df['close'].plot()
#plt.show()


# Set Target Variable
output_var = PD.DataFrame(df['close'])
# Selecting the Features
features = ['open', 'close', 'high', 'low', 'volume']


# Scaling
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(df[features])
feature_transform = PD.DataFrame(columns=features, data=feature_transform, index=df.index)
feature_transform.head()




# Splitting to Training set and Test set
timesplit = TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
    X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index):(len(train_index) + len(test_index))]
    y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index):(len(train_index) + len(test_index))].values.ravel()


trainX = np.array(X_train)
testX = np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])


# Building the LSTM Model
lstm = Sequential()
lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='SGD')
plot_model(lstm, show_shapes=True, show_layer_names=True)




# LSTM Prediction
y_pred= lstm.predict(X_test)




# Predicted vs True Adj Close Value – LSTM
plt.plot(y_test, label='True Value')
plt.plot(y_pred, label='LSTM Value')
plt.title("Prediction by LSTM")
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend()
plt.show()


print('hello world')
