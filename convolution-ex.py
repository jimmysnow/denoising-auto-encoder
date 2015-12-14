import tensorflow as tf
import convolution as cnv
import input_data

# dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Inputs and truth placeholder
x, t = cnv.get_placeholders(784,10)

# ------------------------------------------ #
# Construct network...
# ------------------------------------------ #

# First layer
W_conv1 = cnv.weight_variable([5, 5, 1, 32])
b_conv1 = cnv.bias_variable([32])

# Apply filter
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolve layer with filter
h_conv1 = tf.nn.relu(cnv.conv2d(x_image, W_conv1, 1) + b_conv1)
h_pool1 = cnv.max_pool_2x2(h_conv1)

# ------------------------------------------ #
# Second layer
# ------------------------------------------ #

# Weight & Bias
W_conv2 = cnv.weight_variable([5, 5, 32, 64])
b_conv2 = cnv.bias_variable([64])

# ReLU activation function & Max-pooling layer
h_conv2 = tf.nn.relu(cnv.conv2d(h_pool1, W_conv2, 1) + b_conv2)
h_pool2 = cnv.max_pool_2x2(h_conv2)

# ------------------------------------------ #
# Dense connected layer
# ------------------------------------------ #

# Fully connected layer weights and bias
W_fc1 = cnv.weight_variable([7 * 7 * 64, 1024])
b_fc1 = cnv.bias_variable([1024])

# pooling layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# ------------------------------------------ #
# Output softmax layer
# ------------------------------------------ #

# Weight & Bias
W_fc2 = cnv.weight_variable([1024, 10])
b_fc2 = cnv.bias_variable([10])

# Output Layer
y_conv = cnv.get_softmax_layer(h_fc1_drop, W_fc2, b_fc2)

# ------------------------------------------ #
# Training and Testing functions...
# ------------------------------------------ #

# Cross Entropy loss function
ce = cnv.get_cross_entropy_function(t, y_conv)

# Function to optimize
training_algo = tf.train.AdamOptimizer(1e-4).minimize(ce)

# Correct Prediction
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(t, 1))

# Accuracy calculation
accuracy_calc = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Run the Training...
def run_convolutional_mnist_model(training_algo, accuracy_calc, batch_size):
    config=tf.ConfigProto(allow_soft_placement=True)
    #config.gpu_options.allocator_type = 'BFC'
    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(2000):
            batch = mnist.train.next_batch(batch_size)
            if i % 100 == 0:
                train_accuracy = accuracy_calc.eval(feed_dict={x:batch[0], t:batch[1], keep_prob:1.0})
                print("Epoch: %d, Training Accuracy: %g" % (i, train_accuracy))
            training_algo.run(feed_dict={x:batch[0], t:batch[1], keep_prob:0.5})
            
        test_acc = accuracy_calc.eval(feed_dict={x:mnist.test.images, t:mnist.test.labels, keep_prob:1.0})
        print("Test Accuracy: %g" % test_acc)

# ------------------------------------------ #
# Convolutional MNIST Model -- RUN IT!
# ------------------------------------------ #
batch_size = 100
run_convolutional_mnist_model(training_algo, accuracy_calc, batch_size)


