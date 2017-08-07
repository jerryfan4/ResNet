from Utils import *
import tensorflow as tf

def residual(scope, input_layer, is_training, reuse,
            increase_dim=False, first=False):
    input_dim = input_layer.get_shape().as_list()[-1]

    if increase_dim:
        output_dim = input_dim * 2
        strides = [1, 2, 2, 1]
    else:
        output_dim = input_dim
        strides = [1, 1, 1, 1]

    with tf.variable_scope(scope):
        if first:
            h0 = input_layer
        else:
            h0_bn = batch_norm('h0_bn', input_layer, is_training, reuse)
            h0 = lrelu(h0_bn)

        h1_conv = conv2d('h1_conv', h0, output_dim, strides=strides)
        h1_bn = batch_norm('h1_bn', h1_conv, is_training, reuse)
        h1 = lrelu(h1_bn)

        h2_conv = conv2d('h2_conv', h1, output_dim)
        if increase_dim:
            l = avg_pool('l_pool', input_layer)
            l = tf.pad(l, [[0, 0], [0, 0], 
                           [0, 0], [input_dim // 2, input_dim // 2]])
        else:
            l = input_layer
        h2 = tf.add(h2_conv, l)

        return h2

class ResNet(object):
    def __init__(self, num_units, image_shape,
                 train_batch_size, test_batch_size):

        self.num_units = num_units
        self.image_shape = image_shape
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        [height, width, channels] = image_shape
        train_batch_shape = [train_batch_size, height, width, channels]
        self.train_image_placeholder = tf.placeholder(
            tf.float32,
            shape = train_batch_shape,
            name = 'train_images'
        )
        self.train_label_placeholder = tf.placeholder(
            tf.int32,
            shape = [train_batch_size, ],
            name = 'train_labels'
        )


        test_batch_shape = [test_batch_size, height, width, channels]
        self.test_image_placeholder = tf.placeholder(
            tf.float32,
            shape = test_batch_shape,
            name = 'test_images'
        )
        self.test_label_placeholder = tf.placeholder(
            tf.int32,
            shape = [test_batch_size, ],
            name = 'test_labels'
        )

    def build_network(self, images, is_training, reuse):
        with tf.variable_scope('ResNet', reuse=reuse):
            init_dim = 16
            batch_size = images.get_shape().as_list()[0]

            r0_conv = conv2d('r0_conv', images, init_dim)
            r0_bn = batch_norm('r0_bn', r0_conv, is_training, reuse)
            r0 = lrelu(r0_bn)

            r1_res = residual('res1.0', r0, is_training, reuse, first=True)
            for k in xrange(1, self.num_units):
                r1_res = residual('res1.{}'.format(k), r1_res, is_training, reuse)

            r2_res = residual('res2.0', r1_res, is_training, reuse, increase_dim=True)
            for k in xrange(1, self.num_units):
                r2_res = residual('res2.{}'.format(k), r2_res, is_training, reuse)

            r3_res = residual('res3.0', r2_res, is_training, reuse, increase_dim=True)
            for k in xrange(1, self.num_units):
                r3_res = residual('res3.{}'.format(k), r3_res, is_training, reuse)

            r4_bn = batch_norm('r4_bn', r3_res, is_training, reuse)
            r4 = lrelu(r4_bn)

            axis = [1, 2]
            r5 = tf.reduce_mean(
                r4,
                axis = axis
            )

            fc = fully_connected('fc', tf.reshape(r5, [batch_size, -1]), 10)
            return tf.nn.softmax(fc), fc

    def build_train_op(self):
        train_step = tf.Variable(initial_value=0, trainable=False)
        #train_step_op = train_step.assign_add(1)
        self.train_step = train_step

        prob, logits = self.build_network(self.train_image_placeholder, True, False)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels = self.train_label_placeholder,
            logits = logits
        )
        
        decay = tf.train.exponential_decay(0.0002, train_step, 480000, 0.2, staircase=True)
        decay_loss = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'weight') > 0:
                decay_loss.append(tf.nn.l2_loss(var))
        
        prediction = tf.equal(tf.cast(tf.argmax(prob, axis=1), tf.int32), self.train_label_placeholder)
        self.train_loss = tf.reduce_mean(loss) + tf.mul(decay, tf.add_n(decay_loss))
        self.train_accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

        lr_boundaries = [32000, 48000, 64000]
        lr_values = [0.1, 0.01, 0.001, 0.0002]
        self.learning_rate = tf.train.piecewise_constant(train_step, lr_boundaries, lr_values)

        train_vars = [x for x in tf.trainable_variables() if 'ResNet' in x.name]
        optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(self.train_loss, global_step=train_step, var_list=train_vars)

        return train_op, self.train_loss, self.train_accuracy

    def build_test_op(self):
        prob, logits = self.build_network(self.test_image_placeholder, False, True)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels = self.test_label_placeholder,
            logits = logits
        )
        prediction = tf.equal(tf.cast(tf.argmax(prob, axis=1), tf.int32), self.test_label_placeholder)
        self.test_loss = tf.reduce_mean(loss)
        self.test_accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

        return self.test_loss, self.test_accuracy


