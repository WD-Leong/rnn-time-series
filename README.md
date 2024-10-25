# RNN for Time Series Data

This repository explores the use of a Recurrent Neural Network (RNN) to model time series data. Time series data has its distinct characteristics which includes seasonality and trends. We will simulate a cosine function with a linear trend and model it using RNN. Note that the vanilla RNN is unable to model this simple function with a clear bias in its predictions. To mitigate this, a separate estimate of the intercept needs to be performed and the input sequence to the RNN needs to be modified accordingly to allow the RNN to model this function properly.
