import tensorflow as tf

def initializer(shape, name ='xavier'):
    with tf.variable_scope(name) as scope:
        stddev = 1.0 / tf.sqrt(float(shape[0]), name='stddev')
        inits = tf.truncated_normal(shape=shape, stddev=stddev, name='xavier_init')
    return inits

def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):
    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv

def convolutional_with_stride(input_data, filters_shape, trainable, name, stride=4, activate=True, bn=True):
    with tf.variable_scope(name):
        pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
        paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
        input_data = tf.pad(input_data, paddings, 'CONSTANT')
        strides = (1, stride, stride, 1)
        padding = 'VALID'

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv

def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):
    short_cut = input_data
    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   trainable=trainable, name='conv1')
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1,   filter_num2),
                                   trainable=trainable, name='conv2')

        residual_output = input_data + short_cut

    return residual_output

def resnet_residual(input_data, input_channel, filter_num, trainable, name, downsample=False):
    short_cut = input_data
    with tf.variable_scope(name):
        if downsample:
            input_data = convolutional(input_data, filters_shape=(3, 3, input_channel, filter_num), 
                                       trainable=trainable, downsample=downsample, name='conv1')
            short_cut = convolutional(short_cut, filters_shape=(3,3,input_channel, filter_num), 
                                       trainable=trainable, downsample=downsample, name='sh_conv1')
        else:
            input_data = convolutional(input_data, filters_shape=(3, 3, input_channel, filter_num), trainable=trainable, name='conv_1')
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num, filter_num), activate=False,trainable=trainable, name='conv_2')
        residual_output = input_data + short_cut
    return residual_output

def route(name, previous_output, current_output):
    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)

    return output

def upsample(input_data, name):
    with tf.variable_scope(name):
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2,2), kernel_initializer=tf.random_normal_initializer())
    return output

def dot_product_layer(input, params=None, neurons=1200, name='fc', activation='l_relu'):
    with tf.variable_scope(name) as scope:
        if params is None:
            weights = tf.get_variable(name='weights', initializer=initializer([input.shape[1].value,neurons], name='xavier_weights'))
            bias = tf.get_variable(name='bias', initializer=initializer([neurons], name='xavier_bias'))
        else:
            weights = params[0]
            bias = params[1]

        dot = tf.nn.bias_add(tf.matmul(input, weights, name='dot'), bias, name='pre-activation')
        if activation == 'relu':
            output = tf.nn.relu(dot, name='activity')
        if activation == 'l_relu':
            output = tf.nn.leaky_relu(dot, alpha=0.1, name='activity')
        elif activation == 'sigmoid':
            output = tf.nn.sigmoid(dot, name='activity')            
        elif activation == 'identity':
            output = dot        
    return output

def max_pool_2d_layer(input, pool_size=(1,2,2,1), stride=(1,2,2,1), padding='VALID', name='pool'):    
    with tf.variable_scope(name) as scope:
        output = tf.nn.max_pool(value=input, ksize=pool_size, strides=stride, padding=padding, name=name) 
    return output

def flatten_layer(input, name='flatten'):
    with tf.variable_scope(name) as scope:
        in_shp = input.get_shape().as_list()
        output = tf.reshape(input, [-1, in_shp[1]*in_shp[2]*in_shp[3]])
    return output

def tf_optimizer_func(opt_name):
    if opt_name=="Adam":
        return tf.train.AdamOptimizer
    elif opt_name=="SGD":
        return tf.train.GradientDescentOptimizer
    else:
        raise ValueError("Invalid optimizer name: {}.".format(opt_name))