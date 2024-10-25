# Import the libraries. #
import tensorflow as tf

# Layer Normalization. #
class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, d_model, epsilon=1.0e-3, center=True):
        # center = True will return Layer Normalization, # 
        # center = False will return RMS Normalization.  #
        super(LayerNorm, self).__init__()
        self.center  = center
        self.epsilon = epsilon

        if center:
            self.beta = self.add_weight(
                name="beta", shape=d_model, 
                initializer="zeros", trainable=True)
        else:
            self.beta = 0.0
        
        self.gamma = self.add_weight(
            name="gamma", shape=d_model, 
            initializer="ones", trainable=True)
    
    def call(self, x):
        if self.center:
            x_mean  = tf.reduce_mean(x, axis=-1, keepdims=True)
            x_sigma = tf.math.sqrt(tf.reduce_mean(
                tf.square(x - x_mean), axis=-1, keepdims=True))
            
            x_scale = tf.divide(
                x - x_mean, x_sigma + self.epsilon)
        else:
            x_sigma = tf.math.sqrt(tf.reduce_mean(
                tf.square(x), axis=-1, keepdims=True))
            x_scale = tf.divide(x, x_sigma + self.epsilon)
        
        x_output = self.gamma * x_scale + self.beta
        return x_output

# RNN Layer. #
class RNNLayer(tf.keras.layers.Layer):
    def __init__(
        self, hidden_units, res_conn=True, rate=0.1):
        super(RNNLayer, self).__init__()
        self.rate = rate
        self.res_conn = res_conn
        self.hidden_units = hidden_units

        # RNN weights. #
        self.Wh = tf.keras.layers.Dense(hidden_units)
        self.Wy = tf.keras.layers.Dense(hidden_units)

        # RNN biases. #
        self.bh = self.add_weight(
            "bh", shape=(hidden_units), initializer="zeros")
        self.by = self.add_weight(
            "by", shape=(hidden_units), initializer="zeros")
        
        # Pre-Normalization Layer. #
        #self.lnorm = LayerNorm(hidden_units, epsilon=1.0e-6)

        # Dropout Layer. #
        self.dropout = tf.keras.layers.Dropout(rate)
    
    @tf.function
    def call(
        self, x_curr, h_prev, training=True):
        #x_norm = self.lnorm(x_curr)
        x_conc = tf.concat([x_curr, h_prev], axis=1)
        h_curr = tf.nn.relu(self.Wh(x_conc) + self.bh)
        y_curr = self.Wy(h_curr) + self.by
        
        # Residual Connection. #
        res_output = y_curr
        if self.res_conn:
            res_output += x_curr
        res_output = self.dropout(
            res_output, training=training)
        return (h_curr, res_output)

class RNNNetwork(tf.keras.layers.Layer):
    def __init__(
        self, n_layers, hidden_units, rate=0.1, res_conn=True):
        super(RNNNetwork, self).__init__()
        
        self.rate = rate
        self.n_layers = n_layers
        self.res_conn = res_conn
        self.hidden_units = hidden_units
        
        # Decoder Layers. #
        self.dec_layers = [RNNLayer(
            hidden_units, rate=rate, 
            res_conn=res_conn) for _ in range(n_layers)]
    
    def call(self, x_input, h_prev, training=True):
        h_curr = []
        
        layer_input = x_input
        for m in range(self.n_layers):
            # RNN Layer. #
            output_tuple = self.dec_layers[m](
                layer_input, h_prev[m], training=training)

            h_curr.append(
                tf.expand_dims(output_tuple[0], axis=0))
            
            layer_output = output_tuple[1]
            if self.res_conn:
                layer_output += layer_input
            layer_input = layer_output
        
        rnn_output = layer_output
        h_curr_all = tf.concat(h_curr, axis=0)
        return (h_curr_all, rnn_output)

class RNN(tf.keras.Model):
    def __init__(
        self, n_layers, hidden_units, out_size, 
        max_seq_length, rate=0.1, res_conn=True):
        super(RNN, self).__init__()
        
        self.rate = rate
        self.n_layers = n_layers
        self.res_conn = res_conn
        self.out_size = out_size
        self.max_seq_len = max_seq_length
        self.hidden_units = hidden_units

        # Dropout layer. #
        self.dropout = tf.keras.layers.Dropout(rate)

        # Projection Layers. #
        self.o_proj = tf.keras.layers.Dense(out_size)
        self.i_proj = tf.keras.layers.Dense(hidden_units)
        self.o_bias = self.add_weight(
            "o_bias", shape=(out_size), initializer="zeros")

        # RNN Network. #
        self.rnn_model = RNNNetwork(
            n_layers, hidden_units, 
            rate=rate, res_conn=res_conn)
    
    def call(self, x, h_prev, training=True):
        x_proj = self.i_proj(x)
        x_proj = self.dropout(
            x_proj, training=training)
        
        output_tuple = self.rnn_model(
            x_proj, h_prev, training=training)
        
        h_current  = output_tuple[0]
        dec_output = tf.add(
            self.o_bias, self.o_proj(output_tuple[1]))
        return (h_current, dec_output)

    # For the prefix sum. #
    def forward(self, s_prev, x):
        h_prev = s_prev[0]

        x_proj = self.i_proj(x)
        x_proj = self.dropout(
            x_proj, training=True)
        
        rnn_tuple = self.rnn_model(
            x_proj, h_prev, training=True)
        h_current = rnn_tuple[0]
        return (h_current, rnn_tuple[1])
    
    # Use the prefix sum to compute during training. #
    def decode(self, x, training=True):
        input_shape = x.shape
        batch_size  = input_shape[0]
        zero_shape  = [
            self.n_layers, batch_size, self.hidden_units]
        
        # Initialise the states. #
        h_init = tf.zeros(zero_shape, dtype=tf.float32)
        o_init = tf.zeros(zero_shape[1:], dtype=tf.float32)
        s_init = (h_init, o_init)

        # Reshape the input to seq_len by batch. #
        x_input = tf.transpose(x, [1, 0, 2])
        
        # Run the prefix sum algorithm. #
        rnn_states = tf.scan(
            self.forward, x_input, 
            s_init, parallel_iterations=1)
        
        dec_states  = tf.transpose(
            rnn_states[1], [1, 0, 2])
        dec_outputs = tf.add(
            self.o_bias, self.o_proj(dec_states))
        return dec_outputs
    
    def infer(self, x, gen_len=None):
        input_length = x.shape[1]
        infer_output = [
            tf.expand_dims(x[:, 0, :], axis=1)]
        
        if gen_len is None:
            gen_len = self.max_seq_len
        
        batch_size = tf.shape(x)[0]
        zero_shape = [
            self.n_layers, batch_size, self.hidden_units]
        
        # Initialise the states. #
        h_prev = tf.zeros(zero_shape, dtype=tf.float32)

        for step in range(gen_len):
            curr_inputs = tf.concat(
                infer_output, axis=1)
            next_tuple  = self.call(
                curr_inputs[:, -1, :], 
                h_prev, training=False)
            
            # Update the states. #
            h_prev  = next_tuple[0]
            tmp_out = next_tuple[1]
            
            if step < (input_length-1):
                tmp_in = x[:, step+1, :]
            else:
                tmp_in = tmp_out
            
            infer_output.append(tf.expand_dims(
                tf.cast(tmp_in, tf.double), axis=1))
        return tf.concat(infer_output, axis=1)
