**Apple Stock Market Price Prediction Using LSTM**

Research Question: How can historical stock data from 2010 to 2016 for Apple be used to train a neural network for forecasting future stock prices by implementing a Python script?

Abstract:

This research aims to predict the future stock prices of Apple Inc. by utilizing historical stock market indicators (e.g., open, close, low, high, volume) from 2010 to 2016. Python scripts will be implemented to analyze a comprehensive dataset obtained from the New York Stock Exchange (NYSE) and utilize advanced statistical techniques and machine learning algorithms to build a computationally sophisticated predictive model. The Python scripts will automate the model to achieve efficient and accurate predictions. The developed model will be evaluated using robust metrics such as Mean Squared Error (MSE). Stochastic Gradient Descent (SGD) will be used to train the model. By leveraging these comprehensive categories of variables, this research aims to provide a valuable tool for investors and financial analysts to make informed decisions in the dynamic Apple stock market environment. 

Introduction:

Imagine how lucrative it would be for investors if they were capable of peeking into the future of the stock market, deciphering its mysterious laws, and unraveling the secrets of profitable investing. In today's fast-paced and volatile financial environment, accurately predicting stock market trends is sought after by investors, financial analysts and policy makers. With noise and uncertainty associated with the stock market, manipulating the trend of the stock market, and makes it challenging to identify and predict stock market trends. Market values are affected on a regular basis by a multitude of factors such as changes in the national economy, product values, investor sentiment, weather, and political affairs.[1] For the immense and intricate Apple's market, this study intends to address a fundamental question: how can a neural network be trained using Python scripts to predict future stock prices using Apple's historical stock data from 2010 to 2016, incorporating relevant variables? By building a comprehensive predictive model through this methodology, this research will enable investors and financial analysts to make informed decisions by providing them with a useful and sophisticated tool that will allow them to navigate the dynamic Apple stock market environment with fluidity. By utilizing advanced predictive models with neural networks, investors and analysts can derive in-depth insights into market tendencies, identify potential turning points, and evaluate risk exposures pertaining to particular stocks in a more informed capacity.[2] This research will provide valuable insights for financial analysts to be more precise and effective in evaluating company valuations, providing well-reasoned recommendations to their clients, and ensuring that their investment decisions are commensurate with the dynamics of the market and one's risk profile.[3] In this research, Long Short-Term Memory (LSTM) was utilized to predict the trend of the Apple stock market. The model simulates the volatility of the Apple stock market during the period from 2010 to 2016 using the known Apple stock market, laying the foundation for the Apple stock market in contemporary society and the future, with the aim of making an effective contribution to the field of finance by portfolio management for investors to maximize returns. 

2. Background:

2.1 Neural Networks

A neural network is a computational model consisting of interconnected networks broken into layers.[4] Neural networks are designed to recognize patterns and perform calculations based on input data.[5] Each layer is initiated with weighted connections with bias acting on the lower layers. Each neuron in a layer takes the output from the upper layer as input to that layer and processes the output as input to the lower layer after weighting and biasing.[4] After several more layers of decomposition and narrowing down the computation, a verdict is drawn at the conclusion. The process of training a neural network is called "backpropagation" whereby the difference between the actual output and the desired output, as measured by a cost function such as the Mean Square Error (MSE) loss, is minimized by adjusting the weights and biases of the network.[6] Through innumerable iterations of training, neural networks continuously refine their parameters and learn to make accurate predictions.[7] In stock market forecasting, neural networks play a vital role as they are able to recognize sophisticated patterns and correlations in historical market data. By studying the various factors and indicators that affect stock performance, neural networks can identify relevant features and trends, similar to the way digital components are identified in image recognition tasks.[5] Through backpropagation and gradient descent, neural networks can enhance the accuracy of their predictions by learning from historical stock market data, adjusting to evolving market conditions, and refining their parameters over time.[8] In this way, neural networks can make informed predictions based on its analysis of existing data and extrapolate future stock trends. However, traditional neural network techniques face an inevitable challenge - the gradient explosion and vanishing problem. The gradient explosion and vanishing problems are more likely to occur in multi-layer network training, which is related to the gradient computation during backpropagation, which is crucial for updating the weights and biases of the model.[9] Concerning gradient explosion, the gradient computed during backpropagation will become very large as it is passed back through the layers of the network. This will result in an excessive magnitude of weight updates, which will lead to instability in training and make the model performance erratic. In extreme cases, the gradient will grow exponentially, leading to model divergence.[10]

2.2 Long Short-Term Memory

Long Short-Term Memory (LSTM) is a sort of Recurrent Neural Network (RNN).[8] It takes advantage of a strong nonlinear matching ability between neurons to achieve learning and classification of datasets, and also effectively averts the phenomenon of the gradient explosion/vanishing problem in neural networks.[11] With its unique structure and gating mechanism, we are able to effectively surmount the weakness of gradient explosion/vanishing in neural networks in order to solve the gradient vanishing problem so as to more accurately predict the trend of the stock market. The gradient explosion and vanishing problems may show up during the long serial training of deep networks.[10] The central structure of the LSTM consists of feedback connections that possess a unique memory mechanism that is capable of storing and preserving information over a period of time in the network and classifying it into long-term and short-term memories.[9] It is considered a potential approach to nonlinear modeling because it can extract effective features from large amounts of historical data without any additional knowledge.[11] The ability to capture long-term dependencies and patterns in financial time series data makes LSTMs highly practical in the area of predicting stock prices in terms of performance and accuracy. The key components of LSTM include forgetting gates, input gates, and output gates, which are able to control what information is stored, what new information is added, and what final predictions are outputted, thus ensuring the efficient and stable operation of the system.[12]

1. Forget Gate: The forget gate in an LSTM determines what information from the previous unitary state (long-term memory) should be retained or forgotten. It does this by applying a sigmoid activation function to the combination of the previous hidden state and the current input. The output of the forget gate is a vector of values between 0 and 1 indicating the relevance of the components of the cell state. Elements close to 1 indicate vital information, while elements close to 0 indicate irrelevant information. By using the forget gate, the LSTM can selectively forget or minimize the influence of less relevant information, thus alleviating the gradient vanishing problem.[9][12]

2. Input Gate: The input gate in LSTM serves as a filter that determines which components of the newly computed memory update vector (the result of processing the current input and the previous hidden state) should be added to the output gate state. For this purpose, it employs both the tanh activation function, the sigmoid activation function, and pointwise multiplication to manage the relevance of each component. The tanh activation function is used for the input gate as its output values are in the range [-1, 1], which can be used to identify values that can be coupled to the internal state, which is essential to minimize the influence of certain components in the cell state. This selective update mechanism allows the LSTM to introduce new information into the cell (Long Term Memory) state, acting as a filter that determines which elements of the new memory vectors are worth retaining, and the resulting information will be merged and used to update the cell state while avoiding unstable updates and mitigating the gradient explosion problem.[9][12]

3. Output Gate: The output gate of the LSTM controls which information in the cell state should be included in the new hidden state as the output of the LSTM. The output gate accepts three inputs: the newly updated cell state (long-term memory), the previous hidden state, and the new input data (current information). By using the output gate as a filter, the LSTM can be selective about what information is delivered to the new hidden state in order to avoid passing all the information in the cell state and prevent information from overwhelming the new hidden state. Similar to the forget and input gates, the output gate employs a sigmoid activation function that selectively passes relevant information passed from the tanh activation function. Through this filtering process, the LSTM can selectively output relevant information based on the newly updated cell state, the previous hidden state, and the current input data, ensuring that only relevant information will be passed as output and generate a perfect hidden state. Such selective output of information helps to further mitigate the effects of gradient vanishing or gradient explosion.[9][12]

By incorporating memory cells and gating mechanisms, LSTM can selectively store and access essential historical information, enabling the model to effectively learn long-term dependencies without being affected by the gradient vanishing problem. In addition, LSTM's ability to process the overall data series rather than handling individual points independently makes it capable of taking into account context and dependencies between different time stages, making it ideally suited for enhancing the efficiency and accuracy of stock market forecasts.[13]

3. Data Collection:

Historical stock data for Apple Inc. from 2010 to 2016 was obtained from the dataset available in the Kaggle dataset (https://www.kaggle.com/datasets/dgawlik/nyse?select=fundamentals.csv), an online open-source financial data provider. The dataset contains a range of variables that potentially have an influence on Apple's stock market price. These variables cover financial indicators such as period ending, additional income/expense items, capital expenditures, capital surplus, and cash ratios. In addition, the dataset also contains stock market indicators such as opening price, closing price, minimum price, maximum price and volume. By incorporating these various variables, we are able to conduct a comprehensive analysis that delves into the multiplicity of factors that may be weighing on Apple's stock market price and use such data to speculate on the future trends of Apple's stock market.[14]

4. Implementation:

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

import tensorflow as tf [15]

In this section, we import several Python libraries and modules for performing data manipulation, visualization, machine learning, deep learning, and neural network construction.



* pandas: for data manipulation and analysis.[16]
* numpy: for numerical/mathematical operations. [17]
* pyplot: for creating data visualizations.[18]
* MinMaxScaler: for data scaling.[19]
* Keras.layers: for defining the architecture and functionality of neural network models.[20]
* sklearn.model_selection.TimeSeriesSplit: to provide train/test indices to split time series data samples that are observed at fixed time intervals, in train/test sets.[21]
* sklearn.metrics.mean_squared_error and sklearn.metrics.r2_score: to assess the quality of your predictions. In this case, the program is using mean squared error to evaluate the model performance.[22]
* keras.backend: for executing operations and computations in deep learning models[23]
* keras.callbacks: to perform actions at various stages of training (e.g. at the start or end of an epoch, before or after a single batch, etc).[24][25]
* keras.optimizers.Adam: The Adam optimizer for training neural networks. Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.[26][27]
* keras.utils.plot_model: for plotting and saving the architecture of the model into the file[28]
* tensorflow: a software library for performing numerical calculations using the concept of data flow graphs. In this library, the nodes in the graph represent mathematical operations, and the edges in the graph represent multidimensional arrays of data passed between these nodes, often referred to as "tensors".[29]

tf.config.set_visible_devices([], 'GPU') [15]

This line configures TensorFlow to use only the CPU for computations instead of the GPU. 

whole_data = PD.read_csv("prices-split-adjusted.csv") [15]

This line reads the contents of a CSV file named "prices-split-adjusted.csv" and stores the data in a pandas DataFrame named whole_data. This step loads the entire dataset into memory for further processing.

df = PD.read_csv("prices-split-adjusted.csv", header=0, index_col='date', parse_dates=True) [15]

These lines read and preprocess the data in the "prices-split-adjusted.csv" with indexing and date parsing.  

“header=0” indicates that the first row of the CSV file contains the column names.[30]

“index_col='date'” sets the 'date' column as the index for the dataframe, allowing for time-based indexing and analysis.[31]

“parse_dates=True” tells pandas to interpret the 'date' column as dates, so that they are treated as datetime objects.[32]

target_symbol = 'AAPL'

filtered_df = df[df['symbol'] == target_symbol]

print(filtered_df.head()) [15]

These lines filter data and look for a specific symbol in the whole dataset. The code filters the dataframe df to include only the rows where the 'symbol' column is equal to 'AAPL'. This manipulation produces a new data frame named `filtered_df` containing the stock symbol "AAPL". After that, the “print(filtered_df())” will print the first few lines of the filtered_df data frame. 

print("Filtered Dataframe Shape:", filtered_df.shape) [15]

This line of code outputs the dimensions of the data frame named filtered_df, i.e. the number of rows and columns in the data frame.[33]

print("Null Value Present in Filtered DataFrame:", filtered_df.isnull().values.any()) [15]

This line of code verifies the presence of nulls (missing values) in the `filtered_df` data frame, verifying that each element is null with the `isnull()` function, and then calculating the total number of nulls in each column using `sum()`. Finally, it prints out the corresponding information about the presence of null values in the data frame.[34]

filtered_df['close'].plot()

plt.title(f'{target_symbol} Stock Prices')

plt.xlabel('Date')

plt.ylabel('Close Price')

plt.show() [15]

These lines of code generate a line graph using the ‘close price’ column extracted from the ‘filtered_df’ data frame, which indicates the closing price of the ‘AAPL’ stock. The code also sets the title of the graph to encompass the stock symbol (‘AAPL’) and labels the x-axis with the ‘date’ and the y-axis with the ‘close price’. Finally, it displays the plot using plt.show().

output_var = PD.DataFrame(filtered_df['close']) [15]

This line of code generates a new data frame called "output_var" that is populated exclusively with the "closing price" column extracted from the `filtered_df` data frame. The objective of this measure is to extract the "close_price" column into a separate data frame for subsequent processing and analysis.[35]

features = ['open', 'close', 'high', 'low', 'volume'] [15]

This line defines a list called 'features' that encompasses the column names from the filtered_df DataFrame, specifically selecting columns such as 'open,' 'close,' 'high,' 'low,' and 'volume'. These selected columns are designated as the input variables or features for the LSTM model in subsequent analysis.[36]

scaler = MinMaxScaler()

feature_transform = scaler.fit_transform(filtered_df[features])

feature_transform = PD.DataFrame(columns=features, data=feature_transform, index=filtered_df.index)

feature_transform.head() [15]

These lines utilize the MinMaxScaler to scale selected features within the filtered_df dataframe. By applying this scaling operation, the features are transformed to a common range, between 0 and 1. The scaled values are organized into a new dataframe named feature_transform, where each column corresponds to a scaled feature, and the index is retained from the original filtered_df.[37][38]

`test_index = round(len(feature_transform) * 0.8)` [15]

This line calculates the index that separates the data into training and testing sets. The index is computed as 80% of the total length of the feature_transform dataframe.

`X_train, X_test = feature_transform[:test_index], feature_transform[test_index:]` [15]

This line splits the scaled feature data stored in the feature_transform dataframe into training and testing sets. X_train contains the first test_index rows of scaled features, representing the training set, and X_test contains the remaining rows, representing the testing set.


```
y_train, y_test = output_var[:test_index].values.ravel(), output_var[test_index:].values.ravel()
```


`print(y_train)` [15]

This line splits the output variable into training and testing sets. y_train contains the corresponding values for the training set, and y_test contains the corresponding values for the testing set. The .values.ravel() method is used to convert the pandas Series into a numpy array. The last line prints the array containing the output variable values for the training set.[39]

`naive_predictions = np.roll(y_test, 1)` [15]

This line shifts the y_test array by one position to the right using NumPy's np.roll function. This means that each prediction is set to the value of the previous day's observed value.[40]

`naive_predictions[0] = y_train[-1]` [15]

This line sets the first element in naive_predictions to be the same as the last element in y_test. Since there's no previous day's observed value available in y_test for the first day of the test set, the code sets the first element in naive_predictions to be equal to the last observed value in the training set, which is y_train[-1].


```
naive_mse = mean_squared_error(y_test, naive_predictions)
```


`print("Naive Model MSE:", naive_mse)` [15] [39]

This line calculates the mean squared error between the true test values (y_test) and the predictions made by the naive model (naive_predictions). Then, it prints the calculated mean squared error for the naive model.[41]

trainX = np.array(X_train)

testX = np.array(X_test)

X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])

X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1]) [15] 

These lines convert the training and testing feature datasets from pandas dataframes to numpy arrays. It then reshapes these arrays to match the input shape required by an LSTM model. LSTM models expect input data to be in the form of (batch_size, timesteps, features), where batch_size is the number of samples in each batch, timesteps is the sequence length, and features is the number of input features.[42][43][44][45][46][47][48]

lstm = Sequential()

lstm.add(LSTM(64, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))

lstm.add(Dense(1))

lstm.compile(loss='mean_squared_error', optimizer='adam') [15] 

These lines define the architecture of the LSTM model using Keras' Sequential API. It adds an LSTM layer with 64 units, specifying the input shape (1, trainX.shape[1]) which corresponds to the timesteps and number of features. The activation function used is 'relu', which is an activation function that introduces the property of non-linearity to a deep learning model and solves the vanishing gradients issue. The return_sequences parameter is set to False, indicating that this LSTM layer does not return sequences, instead, the LSTM layer only returns the last output since we only care about the final output. After the LSTM layer, a Dense layer with a single output unit is added, a layer that helps to adjust the dimensionality of the previous layer's output, enabling the model to be more flexible in defining relationships between data.[49][50][51][52][53][54]

plot_model(lstm, show_shapes=True, show_layer_names=True) [15] 

This line generates a visualization of the model architecture using the plot_model function from Keras.

show_shapes=True: allow to get a summary of your model's architecture and the number of parameters that it has/ to show the output shapes of each layer.

show_layer_names=True: allow to show layer names in the graph. [55][56][57]

history=lstm.fit(X_train, y_train, epochs=200, batch_size=1, verbose=1) [15] 

This line trains the compiled LSTM model using the training data (X_train and y_train). 

epochs=200: 200 complete passes of the training dataset through the algorithm.

batch_size=1: samples processed before the model is updated

verbose=1 : Displays progress bar with logs (default).[58][59][60]

y_pred= lstm.predict(X_test) [15] 

This line uses the trained LSTM model to predict the target values for the test data (X_test). The predicted values are stored in the y_pred array.[61]

lstm_mse = mean_squared_error(y_test, y_pred)

print("LSTM MSE: ", lstm_mse) [15] 

This line calculates the mean squared error between the true test values (y_test) and the predictions made by the LSTM model (y_pred). Then, it prints the calculated mean squared error for the LSTM model.[41]

plt.plot(y_test, label='True Value')

plt.plot(y_pred, label='LSTM Value')

plt.plot(naive_predictions, label='Naive Value')

plt.title("Prediction Comparison: LSTM vs Naive")

plt.xlabel('Time Scale')

plt.ylabel('Scaled USD')

plt.legend()

plt.show() [15] 

These lines create a line plot to visualize the comparison between the true values (y_test) and the predicted values (y_pred) obtained from the LSTM model and the naive model. The x-axis represents the time scale, and the y-axis represents the scaled USD values. The legend distinguishes between the true values, LSTM predictions, and naive predictions. The resulting plot helps assess how well the LSTM model performs compared to a simple naive model. plt.legend(): creates an area on the graph which describes all the elements of a graph.[62]

plt.show(): displays the plot.[63]

Section 5: Results

The following table shows the performance of the LSTM model for different unit, epoch and batch size configurations. Each configuration is accompanied by a mean square error value generated during training.


<table>
  <tr>
   <td>configuration
   </td>
   <td>units
   </td>
   <td>epoches
   </td>
   <td>batch size
   </td>
   <td>mean squared error loss
   </td>
  </tr>
  <tr>
   <td>1
   </td>
   <td>32
   </td>
   <td>100
   </td>
   <td>8
   </td>
   <td>0.4104
   </td>
  </tr>
  <tr>
   <td>2
   </td>
   <td>64
   </td>
   <td>100
   </td>
   <td>8
   </td>
   <td>0.4193
   </td>
  </tr>
  <tr>
   <td>3
   </td>
   <td>108
   </td>
   <td>100
   </td>
   <td>8
   </td>
   <td>0.3739
   </td>
  </tr>
  <tr>
   <td>4
   </td>
   <td>108
   </td>
   <td>150
   </td>
   <td>8
   </td>
   <td>0.2362
   </td>
  </tr>
  <tr>
   <td>5
   </td>
   <td>108
   </td>
   <td>150
   </td>
   <td>16
   </td>
   <td>0.3238
   </td>
  </tr>
  <tr>
   <td>6
   </td>
   <td>108
   </td>
   <td>150
   </td>
   <td>6
   </td>
   <td>0.2073
   </td>
  </tr>
  <tr>
   <td>7
   </td>
   <td>108
   </td>
   <td>150
   </td>
   <td>4
   </td>
   <td>0.2475
   </td>
  </tr>
  <tr>
   <td>8
   </td>
   <td>108
   </td>
   <td>150
   </td>
   <td>1
   </td>
   <td>0.0188
   </td>
  </tr>
  <tr>
   <td>9
   </td>
   <td>108
   </td>
   <td>100
   </td>
   <td>1
   </td>
   <td>0.2659
   </td>
  </tr>
  <tr>
   <td>10
   </td>
   <td>32
   </td>
   <td>100
   </td>
   <td>1
   </td>
   <td>0.1822
   </td>
  </tr>
  <tr>
   <td>11
   </td>
   <td>200
   </td>
   <td>100
   </td>
   <td>1
   </td>
   <td>0.0255
   </td>
  </tr>
  <tr>
   <td>12
   </td>
   <td>200
   </td>
   <td>200
   </td>
   <td>1
   </td>
   <td>0.0193
   </td>
  </tr>
  <tr>
   <td>13
   </td>
   <td>108
   </td>
   <td>200
   </td>
   <td>1
   </td>
   <td>0.0135
   </td>
  </tr>
  <tr>
   <td>14
   </td>
   <td><strong>64</strong>
   </td>
   <td><strong>200</strong>
   </td>
   <td><strong>1</strong>
   </td>
   <td><strong>0.0085</strong>
   </td>
  </tr>
  <tr>
   <td>15
   </td>
   <td>32
   </td>
   <td>200
   </td>
   <td>1
   </td>
   <td>0.0150
   </td>
  </tr>
</table>


Naive Model MSE: 2.9258174918288593

LSTM Model MSE: 0.0085


![alt_text](images/image1.png "image_tooltip")


The evaluation process of the prediction model is conducted by using the Mean Squared Error (MSE) as a measure of performance. As it quantifies the prediction error of a model, it enables us to comply with comparing the performance between different models and the distance between predicted and real values.[64] For a better assessment and analysis of the performance of LSTM models in forecasting, we propose a simple benchmark model, often referred to as the " naive model". In time series forecasting, the naive model employs some basic and intuitive forecasting methods, such as the assumption that the predicted value at time "t" is equal to the actual value at time "t-1" for forecasting purposes.[65][66] By introducing the naive model, we can build a straightforward and intuitive benchmark for evaluating the performance of the LSTM model to obtain a better understanding of the effectiveness of the LSTM model in predicting future stock movements. Specifically, we will calculate the mean square error (MSE) of the LSTM model and the naive model and compare them in performance. If the MSE of the LSTM model is significantly lower than the naive model, then we can conclude that the LSTM model performs better on more sophisticated issues in stock price prediction. This comparison helps to ensure that our models have practical predictive power in real-world applications.[67] Ultimately, naive models offer not only a basic performance benchmark for evaluating more advanced statistical and machine learning models, but also serve as a rapid prototyping tool for exploring data, laying the groundwork for more complex modeling work, and offering initial insights into the patterns of the data and the accuracy of the predictions.[68] As per the results, the Naive model has a mean squared error of 2.9258, which means that it has a large deviation from the actual price. The high MSE suggests that the Naive model has a limited capability to comprehend the underlying trend and hence its prediction reliability is relatively low. Contrarily, the LSTM model has a notably lower mean squared error loss value of 0.0085. This remarkable difference highlights the stronger predictive ability of the LSTM model relative to the Naive model. We provide details on the configuration of the LSTM model training, highlighting the number of units, the epoch duration, the batch size, and the corresponding mean squared error loss values for different settings. Reportedly, the performance of the LSTM model is dramatically improved by adjusting the epoch and batch sizes. In the preliminary case, training with 32 units, 100 epochs and 8 batches yields a mean squared error value loss of 0.4104. Through continuous parameter tuning, we succeeded in gradually reducing the mean square error (MSE) loss value and finally found the optimal model configuration: 108 units, 150 epochs and 1 batch. This process ultimately reduces the mean squared error loss to 0.0085. The reduction in the mean square error loss value of the LSTM model indicates that it is more capable of modeling the trend of the actual stock prices, implying that it is expected to provide more accurate prediction and capture the dynamics of minor differences in the market. These results amply demonstrate the close relationship between the accuracy of LSTM models and parameter optimization. Increasing the number of neuron units and the number of epochs typically enhances the model performance, while decreasing the batch size contributes to the accuracy of the model. Based on the mean square error loss values we obtained, the predictions of the model are closely related to the actual stock prices, which makes it a practical tool for investors and financial analysts. In conclusion, our study emphasizes the importance of parameter tuning to improve forecasting accuracy by exploring different system configurations. The LSTM model performs well in this study with a minimum mean square error loss value of 0.0085, reaffirming its potential as a potent tool for forecasting stock prices. This renders the LSTM model a conducive tool for making informed decisions in a sophisticated stock market environment.

Section 6: Conclusion

In this research, we successfully addressed a fundamental research question on how to train a neural network model with the capability of forecasting future stock prices using Python scripts combined with Apple's historical stock data from 2010 to 2016 along with a variety of relevant variables. Our primary objective is to offer investors and financial analysts an effective tool that will enable them to make informed decisions in the evolving Apple stock market environment. Our implementation process covers complex steps of data collection, preprocessing, and model training. In this study, we leverage the strength of neural networks, specifically the Long Short-Term Memory architecture, to demonstrate the potential of machine learning in capturing sophisticated stock market dynamics. We obtained historical stock data from the Kaggle dataset, which contains various variables of stock market indicators. We trained and fine tuned the LSTM model in elaborate stages with meticulous parameter tuning. In the results section, we vividly demonstrate the performance variation of the LSTM model under different configurations. By comparing the Naive model with the LSTM model, we highlight the outstanding prediction ability of the LSTM model. It demonstrates its preeminence by dramatically reducing the Mean Squared Error. After systematically exploring the number of cells, the epoch, and batch size, we demonstrate how the accuracy of the model progressively improves with parameter finetuning. Eventually, we found the optimal settings, i.e., using 64 units, 200 epochs, and 1 batch size, which reduced the minimum mean squared error loss of the LSTM model to 0.0085. This again validates the ability of the LSTM model in recognizing complex patterns and accurately predicting stock prices. In conclusion, this study successfully addressed the research questions, achieved our research objectives, and demonstrated the power of neural networks (especially LSTM) in predicting stock prices. Through the implementation of Python scripts, we ingeniously used historical stock data to construct advanced LSTM models that outperform the base Naive model dramatically. Our results highlight the potential of LSTM models to provide investors and financial analysts with a valuable tool in the volatile Apple stock market. By providing more accurate predictions and minimizing risk, the model makes an essential contribution to the dynamic field of stock trading.

7. Sources:

[1]Khaidem, Luckyson, et al. “Predicting the Direction of Stock Market Prices Using Random Forest.” Applied Mathematical Finance, vol. 00, no. 00, 2016, pp. 1–20, arxiv.org/pdf/1605.00003.pdf.

[2] “Advantages of Stock Market Prediction | Benefits You Must Know.” Stock Pathshala, 3 June 2022, www.stockpathshala.com/advantages-of-stock-market-prediction/#:~:text=Stock%20market%20prediction%20helps%20you. Accessed 1 Sept. 2023.

[3] ​​Zucchi, Kristina. “Stock Analysis: Forecasting Revenue and Growth.” Investopedia, 1 Dec. 2021, www.investopedia.com/articles/active-trading/022315/stock-analysis-forecasting-revenue-and-growth.asp.

[4]3Blue1Brown. “But What Is a Neural Network? | Deep Learning, Chapter 1.” YouTube, 5 Oct. 2017, www.youtube.com/watch?v=aircAruvnKk (“But what is a neural network? | Chapter 1, Deep learning”).

[5] Zhou, Victor. “Machine Learning for Beginners: An Introduction to Neural Networks.” Medium, 20 Dec. 2019, towardsdatascience.com/machine-learning-for-beginners-an-introduction-to-neural-networks-d49f22d238f9.

[6] 3Blue1Brown. “Gradient Descent, How Neural Networks Learn | Deep Learning, Chapter 2.” YouTube, 16 Oct. 2017, www.youtube.com/watch?v=IHZwWFHWa-w.

[7] 3Blue1Brown. “What Is Backpropagation Really Doing? | Deep Learning, Chapter 3.” YouTube, 3 Nov. 2017, www.youtube.com/watch?v=Ilg3gGewQ5U.

[8] Pawar, Kriti, et al. “Stock Market Price Prediction Using LSTM RNN.” SpringerLink, 1 Jan. 1970, link.springer.com/chapter/10.1007/978-981-13-2285-3_58.

[9] Dolphin, Rian. “LSTM Networks | a Detailed Explanation.” Medium, 26 Mar. 2021, towardsdatascience.com/lstm-networks-a-detailed-explanation-8fae6aefc7f9.

[10] Bohra, Yash. “Vanishing and Exploding Gradients in Deep Neural Networks.” Analytics Vidhya, 18 June 2021, www.analyticsvidhya.com/blog/2021/06/the-challenge-of-vanishing-exploding-gradients-in-deep-neural-networks/.

[11] Shah, Jaimin, et al. “A Comprehensive Review on Multiple Hybrid Deep Learning Approaches for Stock Prediction.” Intelligent Systems with Applications, vol. 16, Nov. 2022, p. 200111, https://doi.org/10.1016/j.iswa.2022.200111.

[12] “Long Short-Term Memory (LSTM), Clearly Explained.” Www.youtube.com, www.youtube.com/watch?v=YCzL96nL7j0.

[13] Keith, Michael. “Exploring the LSTM Neural Network Model for Time Series.” Medium, 7 Oct. 2022, towardsdatascience.com/exploring-the-lstm-neural-network-model-for-time-series-8b7685aa8cf.

[14] “New York Stock Exchange.” Www.kaggle.com, www.kaggle.com/datasets/dgawlik/nyse?resource=download&%3Bselect=prices-split-adjusted.csv. Accessed 1 Sept. 2023.

[15]Sharma, Prashant. “Stock Market Prediction | Machine Learning for Stock Market Prediction.” _Analytics Vidhya_, 13 Oct. 2021, [www.analyticsvidhya.com/blog/2021/10/machine-learning-for-stock-market-prediction-with-step-by-step-implementation/](www.analyticsvidhya.com/blog/2021/10/machine-learning-for-stock-market-prediction-with-step-by-step-implementation/).

[16]“Pandas Introduction.” Www.w3schools.com, www.w3schools.com/python/pandas/pandas_intro.asp#:~:text=Pandas%20is%20a%20Python%20library.

[17] Numpy. “NumPy: The Absolute Basics for Beginners — NumPy V1.20 Manual.” Numpy.org, numpy.org/doc/stable/user/absolute_beginners.html.

[18] “What Is Pyplot in Python?” Educative: Interactive Courses for Software Developers, www.educative.io/answers/what-is-pyplot-in-python. Accessed 1 Sept. 2023.

[19]Kumar, Ajitesh. “MinMaxScaler vs StandardScaler - Python Examples.” Data Analytics, 13 Apr. 2023, vitalflux.com/minmaxscaler-standardscaler-python-examples/#:~:text=MinMaxScaler%20is%20useful%20when%20the.

[20] “What Are Keras Layers?” Educative: Interactive Courses for Software Developers, www.educative.io/answers/what-are-keras-layers. Accessed 1 Sept. 2023.

[21]“Sklearn.model_selection.TimeSeriesSplit.” Scikit-Learn, scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html#:~:text=Provides%20train%2Ftest%20indices%20to. Accessed 1 Sept. 2023.

[22] “What Are Sklearn Metrics and Why You Need to Know about Them?” UpGrad Blog, www.upgrad.com/blog/what-are-sklearn-metrics/.

[23] “What Is Keras Backend?” Educative: Interactive Courses for Software Developers, www.educative.io/answers/what-is-keras-backend. Accessed 2 Sept. 2023.

[24]Team, Keras. “Keras Documentation: Callbacks API.” Keras.io, keras.io/api/callbacks/#:~:text=A%20callback%20is%20an%20object. Accessed 2 Sept. 2023.

[25]_Monkeypox Detection Using Computer Vision in Python - Wisdom ML_. 1 Aug. 2022, wisdomml.in/monkeypox-detection-using-computer-vision-in-python/. Accessed 5 Sept. 2023.

[26]“Tf.keras.optimizers.legacy.Adam | TensorFlow V2.13.0.” TensorFlow, www.tensorflow.org/api_docs/python/tf/keras/optimizers/legacy/Adam#:~:text=Adam%20optimization%20is%20a%20stochastic. Accessed 2 Sept. 2023..

[27]“Facial Expression Recognition Projc 2 (3) (1).” Www.slideshare.net, www.slideshare.net/AbhiAchalla/facial-expression-recognition-projc-2-3-1. Accessed 5 Sept. 2023.

[28] “Keras Plot Model | How to Plot Model Architecture in Keras?” EDUCBA, 4 Oct. 2022, www.educba.com/keras-plot-model/. Accessed 2 Sept. 2023.

[29] “Introduction to TensorFlow.” GeeksforGeeks, 4 Aug. 2017, www.geeksforgeeks.org/introduction-to-tensorflow/.

[30]H, Barney. “How to Read CSV File into Python Using Pandas.” Medium, 3 June 2020, towardsdatascience.com/how-to-read-csv-file-using-pandas-ab1f5e7e7b58#:~:text=Default%20value%20is%20header%3D0. Accessed 2 Sept. 2023.

[31]H, Barney. “How to Read CSV File into Python Using Pandas.” Medium, 3 June 2020, towardsdatascience.com/how-to-read-csv-file-using-pandas-ab1f5e7e7b58#:~:text=index_col%3A%20This%20is%20to%20allow. Accessed 2 Sept. 2023.

[32]Chen, B. “4 Tricks You Should Know to Parse Date Columns with Pandas Read_csv().” Medium, 28 Jan. 2021, towardsdatascience.com/4-tricks-you-should-know-to-parse-date-columns-with-pandas-read-csv-27355bb2ad0e.

[33] “How to Return the Shape of a DataFrame in Pandas.” Educative: Interactive Courses for Software Developers, www.educative.io/answers/how-to-return-the-shape-of-a-dataframe-in-pandas.

[34]“Null in Python: How to Set None in Python? (with Code).” FavTutor, favtutor.com/blogs/null-python#:~:text=Python%20doesn. Accessed 2 Sept. 2023.

[35]Sharma, Prashant. “Stock Market Prediction | Machine Learning for Stock Market Prediction.” Analytics Vidhya, 13 Oct. 2021, www.analyticsvidhya.com/blog/2021/10/machine-learning-for-stock-market-prediction-with-step-by-step-implementation/.

[36]“Create a Pandas DataFrame from Lists.” GeeksforGeeks, 17 Dec. 2018, [www.geeksforgeeks.org/create-a-pandas-dataframe-from-lists/](www.geeksforgeeks.org/create-a-pandas-dataframe-from-lists/).

[37]Sharma, Prashant. “Stock Market Prediction | Machine Learning for Stock Market Prediction.” Analytics Vidhya, 13 Oct. 2021, www.analyticsvidhya.com/blog/2021/10/machine-learning-for-stock-market-prediction-with-step-by-step-implementation/.

[38]“StandardScaler, MinMaxScaler and RobustScaler Techniques - ML.” GeeksforGeeks, 15 July 2020, www.geeksforgeeks.org/standardscaler-minmaxscaler-and-robustscaler-techniques-ml/.

[39]“Numpy.ravel() in Python - Javatpoint.” Www.javatpoint.com, www.javatpoint.com/numpy-ravel. Accessed 2 Sept. 2023.

[40]“NumPy Roll().” Www.programiz.com, www.programiz.com/python-programming/numpy/methods/roll#:~:text=The%20roll()%20method%20allows. Accessed 2 Sept. 2023.

[41] “LSTM Time Series + Stock Price Prediction = FAIL.” _Kaggle.com_, www.kaggle.com/code/carlmcbrideellis/lstm-time-series-stock-price-prediction-fail. Accessed 2 Sept. 2023.

[42]“Convert Python List to Numpy Arrays.” GeeksforGeeks, 10 Feb. 2020, www.geeksforgeeks.org/convert-python-list-to-numpy-arrays/.

[43]“How to Calculate Mean Squared Error in Python • Datagy.” Datagy, 10 Jan. 2022, datagy.io/mean-squared-error-python/.

[44]“Numpy.reshape — NumPy V1.20 Manual.” Numpy.org, numpy.org/doc/stable/reference/generated/numpy.reshape.html.

[45]“Fit the LSTM Model in Python Using Keras: A Comprehensive Guide for Data Scientists | Saturn Cloud Blog.” Saturncloud.io, 10 July 2023, saturncloud.io/blog/fit-the-lstm-model-in-python-using-keras-a-comprehensive-guide-for-data-scientists/. Accessed 2 Sept. 2023.

[46]“Understanding Keras LSTM Input Shape for Data Scientists | Saturn Cloud Blog.” Saturncloud.io, 10 July 2023, saturncloud.io/blog/understanding-keras-lstm-input-shape-for-data-scientists/#:~:text=Remember%2C%20the%20input%20shape%20for. Accessed 2 Sept. 2023.

[47]“Understanding Keras LSTMs: Role of Batch-Size and Statefulness | Saturn Cloud Blog.” Saturncloud.io, 10 July 2023, saturncloud.io/blog/understanding-keras-lstms-role-of-batchsize-and-statefulness/#:~:text=Understanding%20Batch%2Dsize%20in%20Keras. Accessed 2 Sept. 2023.

[48]Bikmukhametov, Timur. “How to Reshape Data and Do Regression for Time Series Using LSTM.” Medium, 13 Apr. 2020, towardsdatascience.com/how-to-reshape-data-and-do-regression-for-time-series-using-lstm-133dad96cd00. Accessed 2 Sept. 2023.

[49]“Difference between Samples, Time Steps and Features in Neural Network.” Cross Validated, stats.stackexchange.com/questions/264546/difference-between-samples-time-steps-and-features-in-neural-network. Accessed 2 Sept. 2023.

[50]“Keras Input Explanation: Input_shape, Units, Batch_size, Dim, Etc.” Stack Overflow, stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc#:~:text=The%20input%20shape&text=It. Accessed 2 Sept. 2023.

[51]“ReLU Activation Function Explained | Built In.” Builtin.com, builtin.com/machine-learning/relu-activation-function.

[52]“Understanding Keras LSTMs: Clearing Common Doubts | Saturn Cloud Blog.” Saturncloud.io, 10 July 2023, saturncloud.io/blog/understanding-keras-lstms-clearing-common-doubts/#:~:text=This%20is%20useful%20when%20stacking. Accessed 2 Sept. 2023.

[53]“TensorFlow Dense | How to Use Function Tensorflow Dense?” EDUCBA, 13 July 2022, www.educba.com/tensorflow-dense/. Accessed 2 Sept. 2023.

[54]Verma, Yugesh. “A Complete Understanding of Dense Layers in Neural Networks.” Analytics India Magazine, 19 Sept. 2021, analyticsindiamag.com/a-complete-understanding-of-dense-layers-in-neural-networks/.

[55]Fares, Anass El. “A Better Understanding How a CNN Works (Code Available).” Medium, 30 Dec. 2021, medium.com/@El_Fares_Anass/a-better-understanding-how-a-cnn-works-code-available-e880f9a338dc.

‌

[56]“What Is Tensorflow Show Model Summary | Saturn Cloud Blog.” Saturncloud.io, 13 June 2023, saturncloud.io/blog/what-is-tensorflow-show-model-summary/#:~:text=Tensorflow%20show%20model%20summary%20is%20a%20function%20that%20allows%20you. Accessed 2 Sept. 2023.

[57]Brownlee, J. (2019) How to visualize a deep learning neural network model in Keras, MachineLearningMastery.com. Available at: https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/ (Accessed: 01 September 2023). 

[58]“Deep Learning Tips and Tricks.” KDnuggets, www.kdnuggets.com/2018/07/deep-learning-tips-tricks.html. Accessed 2 Sept. 2023.

[59]“Understanding the Use of Verbose in Keras Model Validation | Saturn Cloud Blog.” Saturncloud.io, 10 July 2023, saturncloud.io/blog/understanding-the-use-of-verbose-in-keras-model-validation/#:~:text=The%20verbose%20argument%20can%20take. Accessed 2 Sept. 2023.

[60]“What Is Epoch in Machine Learning?| UNext.” UNext, 24 Nov. 2022, u-next.com/blogs/machine-learning/epoch-in-machine-learning/#:~:text=An%20epoch%20in%20machine%20learning. Accessed 2 Sept. 2023.

[61]Brownlee, Jason. “Difference between a Batch and an Epoch in a Neural Network.” Machine Learning Mastery, 9 Aug. 2022, machinelearningmastery.com/difference-between-a-batch-and-an-epoch/#:~:text=The%20batch%20size%20is%20a%20number%20of%20samples%20processed%20before.

[62]Python Predict() Function - All You Need to Know! - AskPython. 13 Oct. 2020, www.askpython.com/python/examples/python-predict-function.

[63]“Explain What Are Legends with an Example Using Matplotlib ? -.” ProjectPro, www.projectpro.io/recipes/explain-what-are-legends-with-example-matplotlib#:~:text=A%20legend%20is%20a%20predefined.

[64]“Matplotlib.pyplot.show() in Python.” GeeksforGeeks, 1 Apr. 2020, www.geeksforgeeks.org/matplotlib-pyplot-show-in-python/. Accessed 2 Sept. 2023.

[65]Ghorbani, Mahsa, and Edwin K. P. Chong. “Stock Price Prediction Using Principal Components.” _PLOS ONE_, vol. 15, no. 3, 20 Mar. 2020, p. e0230124, https://doi.org/10.1371/journal.pone.0230124.

[66]“Time Series - Naive Methods.” _Www.tutorialspoint.com_, www.tutorialspoint.com/time_series/time_series_naive_methods.htm#:~:text=Naive%20Methods%20such%20as%20assuming. Accessed 2 Sept. 2023.

[67]“Forecasting Methods: Naive Forecasting.” _Www.avercast.com_, [www.avercast.com/post/naive-forecasting](www.avercast.com/post/naive-forecasting).

‌[68]Howell, Egor. “Basic Time Series Forecasting Techniques.” _Medium_, 28 Dec. 2022, towardsdatascience.com/basic-forecasting-techniques-ef4295248e46.

