import tensorflow as tf

def conv2d(scope, input_layer, output_dim, use_bias=False,
            filter_size=3, strides=[1, 1, 1, 1]):
    input_dim = input_layer.get_shape().as_list()[-1]

    with tf.variable_scope(scope):
        conv_filter = tf.get_variable(
            'conv_weight',
            shape = [filter_size, filter_size, input_dim, output_dim],
            dtype = tf.float32,
            initializer = tf.contrib.layers.variance_scaling_initializer(),
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)
        )
        conv = tf.nn.conv2d(input_layer, conv_filter, strides, 'SAME')

        if use_bias:
            bias = tf.get_variable(
                'conv_bias',
                shape = [output_dim],
                dtype = tf.float32,
                initializer = tf.constant_initializer(0.0)
            )

            output_layer = tf.nn.bias_add(conv, bias)
            output_layer = tf.reshape(output_layer, conv.get_shape())
        else:
            output_layer = conv

        return output_layer

def batch_norm(scope, input_layer, is_training, reuse):
    '''
    output_layer = tf.contrib.layers.batch_norm(
        input_layer,
        decay = 0.9,
        scale = True,
        epsilon = 1e-5,
        is_training = is_training,
        reuse = reuse,
        scope = scope
    )

    '''
    with tf.variable_scope(scope, reuse=reuse):
        input_dim = input_layer.get_shape().as_list()[-1]
        mean, variance = tf.nn.moments(input_layer, [0, 1, 2])
        beta = tf.get_variable(
            'bn_beta',
            shape = [input_dim],
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.0)
        )
        gamma = tf.get_variable(
            'bn_gamma',
            shape = [input_dim],
            dtype = tf.float32,
            initializer = tf.constant_initializer(1.0)
        )

        output_layer = tf.nn.batch_normalization(input_layer, mean, variance,
                                                 beta, gamma, 0.00001)

    return output_layer

def lrelu(input_layer, leak=0.2):
    output_layer = tf.nn.relu(input_layer)
    #output_layer = tf.maximum(input_layer, leak * input_layer)
    return output_layer

def fully_connected(scope, input_layer, output_dim):
    input_dim = input_layer.get_shape().as_list()[-1]
    
    with tf.variable_scope(scope):
        fc_weight = tf.get_variable(
            'fc_weight',
            shape = [input_dim, output_dim],
            dtype = tf.float32,
            initializer = tf.contrib.layers.variance_scaling_initializer(),
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)            
        )

        fc_bias = tf.get_variable(
            'fc_bias',
            shape = [output_dim],
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.0)
        )

        output_layer = tf.matmul(input_layer, fc_weight) + fc_bias

        return output_layer

def avg_pool(scope, input_layer, ksize=None, strides=[1, 2, 2, 1]):
    if ksize is None:
        ksize = strides

    with tf.variable_scope(scope):
        output_layer = tf.nn.avg_pool(input_layer, ksize, strides, 'VALID')
        return output_layer


