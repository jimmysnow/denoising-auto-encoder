import tensorflow as tf

# --------------------- #
#  convolution library  #
# --------------------- #

# Input & Truth Vector
def get_placeholders(input_dim, truth_dim):
    x = tf.placeholder("float",shape=[None,input_dim])
    t = tf.placeholder("float",shape=[None,truth_dim])
    return (x, t)

# Weights & Bias
def get_model_params(input_dim, truth_dim, bias_dim):
    W = tf.Variable(tf.zeros([input_dim,truth_dim]))
    b = tf.Variable(tf.zeros([bias_dim]))
    return (W, b)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, shift):
    return tf.nn.conv2d(x, W, strides=[shift, shift, shift, shift], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_nxm(x, n, m):
    return tf.nn.max_pool(x, ksize=[1, n, m, 1], strides=[1, n, m, 1], padding='SAME')

# Softmax Layer
def get_softmax_layer(x, W, b):
    return tf.nn.softmax(tf.matmul(x, W) + b)

# Cost Function
def get_cross_entropy_function(t, y):
    return -tf.reduce_sum(t * tf.log(y))

# Training module
def get_training_module(learning_rate, cost_function):
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Test the model
def do_test_model(inputs, outputs, truth):
    correct_prediction = tf.equal(tf.argmax(outputs,1), tf.argmax(truth,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    test_accuracy = accuracy.eval(feed_dict={inputs: mnist.test.images, truth: mnist.test.labels})
    print("Test Accuracy: " + str(test_accuracy))

# Training iterations
def do_train_model(dataset, training_algo, input_values, truth_values, batch_size, num_epochs):
    for i in range(num_epochs):
        if i % (num_epochs/10) == 0: print("Epoch " + str(i) + "...")
        batch = dataset.train.next_batch(batch_size)
        training_algo.run(feed_dict={input_values: batch[0], truth_values: batch[1]})

# Train/test
def do_train_test(learning_rate, batch_size, num_epochs):
    x, t = get_placeholders()
    W, b = get_model_params()
    y = get_softmax_layer(x, W, b)
    ce = get_cross_entropy_function(t, y)
    training_algo = get_training_module(learning_rate, ce)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        do_train_model(training_algo, x, t, batch_size, num_epochs)
        do_test_model(x, y, t)

