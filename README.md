# RNN for Time Series Data

This repository explores the use of a Recurrent Neural Network (RNN) to model time series data. Time series data has its distinct characteristics which includes seasonality and trends. We will simulate a cosine function with a linear trend and model it using RNN. Note that the vanilla RNN is unable to model this simple function with a clear bias in its predictions. To mitigate this, a separate estimate of the intercept needs to be performed and the input sequence to the RNN needs to be modified accordingly to allow the RNN to model this function properly.

Without the intercept estimation, the vanilla RNN will show a residual bias in its prediction:

![screenshot_1](vanilla_rnn.png)
The notebook to reference is `rnn_trial_v0.ipynb`.

However, by incorporating the estimate of the intercept, the RNN is able to predict the trend and seasonality properly:

![screenshot_2](rnn_with_intercept.png)
The notebook to reference is `rnn_trial_tsa.ipynb`.

## Miscellaneous Information

The RNN module `tf_ver2_rnn_tsa_scan.py` uses Tensorflow's `tf.scan` function to run the RNN. It will convert the base RNN layer into a Tensorflow graph via the `tf.function` wrapper to improve its runtime. The use of the `tf.scan` functionality allows the RNN to be able to balance between modeling long sequentual data with a relatively longer runtime at a modest memory footprint. 
