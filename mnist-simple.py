import tensorflow as tf
import input_data # comes from the file provided in the tutorial...


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # mnist.train -- training data
    # mnist.test -- testing data
    # mnist.train.images -- training images
    # mnist.train.labels -- training labesls

    print("Here's what the training data looks like:\n")
    print(mnist.train.images)
    print("\nLen images: " + str(len(mnist.train.images)))
    print("Len labels: " + str(len(mnist.train.labels)))

    # create an input vector for flattened images...
    x = tf.placeholder(tf.float32, [None, 784])

    # weight matrix 784 x 10 
    W = tf.Variable(tf.zeros([784,10]))

    # Biases
    b = tf.Variable(tf.zeros([10]))
    
    # y = output = softmax(Sum(W * x) + b)
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    
    # Truth value
    t = tf.placeholder(tf.float32, [None, 10])
    
    # Loss function
    cross_entropy = -tf.reduce_sum(t * tf.log(y))
    
    # GD optimization
    learning_rate = 0.01
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    
    # initialize stuff. duh.
    init = tf.initialize_all_variables()

    # create a session
    sess = tf.Session()

    # Run the session
    sess.run(init)
    
    # 1000 training iterations
    batch_size = 100
    num_epochs = 1000

    for i in range(num_epochs):
        # periodic print out
        if i % (100) == 0: print("Epoch: " + str(i) + "...")
        batch_inputs, truth_values = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_inputs, t: truth_values})
        
    # compare truth values to predictions
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))

    # calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # print the accuracy
    print("Accuracy on Test set: " + str(sess.run(accuracy, feed_dict={x: mnist.test.images, t: mnist.test.labels})))
    
if __name__ == "__main__": main()
