import tensorflow as tf


def conv2d(x, W):
    # input shape assumes NHWC
    # weight shape assumes HWC(in)C(out)
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool(x, ksize=2, stride=2):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1], padding='SAME')


def weight_variable(shape, stdev=0.1):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape, init=0.1):
    initial = tf.constant(init, shape=shape)
    return tf.Variable(initial)


class CNN(object):
    def __init__(self, input, keep_prob):
        # assuming input as [N, 28, 28], the conv1 has [N, 28, 28, 32]
        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])
        self.h_conv1 = tf.nn.relu(conv2d(input, self.W_conv1) + self.b_conv1)

        # the pool1 has [N, 14, 14, 32]
        self.h_pool1 = maxpool(self.h_conv1)

        # the conv2 has [N, 14, 14, 32]
        self.W_conv2 = weight_variable([5, 5, 32, 32])
        self.b_conv2 = bias_variable([32])
        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)

        # the pool2 has [N, 7, 7, 32]
        self.h_pool2 = maxpool(self.h_conv2)

        # classifier part, flatten h_pool2 is the input
        self.W_fc1 = weight_variable([7*7*32, 10])
        self.b_fc1 = bias_variable([10])

        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*32])
        self.keep_prob = tf.placeholder(tf.float32)
        self.h_pool2drop = tf.nn.dropout(self.h_pool2_flat, keep_prob)
        self.y_conv = tf.nn.relu(tf.matmul(self.h_pool2drop, self.W_fc1) + self.b_fc1)

    def output(self):
        return self.y_conv

    def globalcost(self, teacher, student):
        self.x_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=teacher, logits=student))
        return self.x_entropy


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 28*28])
    y = tf.placeholder(tf.float32, [None, 10])
    keeprate = tf.placeholder(tf.float32)
    x_ = tf.reshape(x, [-1, 28, 28, 1])
    cnn = CNN(x_, keeprate)
    y_ = cnn.output()

    cost = cnn.globalcost(y, y_)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

    correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    for i in range(20000):
        bat_xs, bat_ys = mnist.train.next_batch(50)
        if i % 100 == 0:
            tr_acc = acc.eval(session=sess, feed_dict={x: bat_xs, y: bat_ys, keeprate: 1.0})
            print "step %05d: train acc = %.4g" % (i, tr_acc)
        sess.run(train_step, feed_dict={x: bat_xs, y: bat_ys, keeprate: 0.5})

    print '---eval Prediction error---'
    print(sess.run(acc, feed_dict={x: mnist.test.images, y: mnist.test.labels, keeprate: 1.0}))
