# Import TensorFlow-Library
import tensorflow as tf

# Import DateTime for recording training time
from datetime import datetime

# Import MNIST DATA
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Setting Parameters for Network
learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 1

# Building up Network Input
x = tf.placeholder("float", [None, 784], name="Input")
y = tf.placeholder("float", [None, 10], name="Output")

# Creating a model

# Setting model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

with tf.name_scope("Wx_b") as scope:
    # Constructing a linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b)

# Collecting Data
w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)

# Building more scopes

with tf.name_scope("cost_function") as scope:
    # Minimizing the cost function using cross entropy
    cost_function = -tf.reduce_sum(y*tf.log(model))
    # Creating a summary representing the cost function
    tf.summary.scalar("cost_function", cost_function)

with tf.name_scope("train") as scope:
    # Gradient descent training
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Initializing variables
init = tf.global_variables_initializer()

# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()

# Launching the graph
with tf.Session() as sess:
    sess.run(init)

    # Setting the logs writer to the folder /tmp/tensorflow_logs
    summary_writer = tf.summary.FileWriter('C:\TensorFlow\logs', sess.graph)

    # Setting time before training
    timeBeforeTraining = datetime.now()

    # Training cycle
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # a loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Computing the average loss
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            # Writing logs for each iteration
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch + i)
        # Displaying logs per iteration
        if iteration % display_step == 0:
            print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

    # Setting time after Training and calculate the difference
    timeAfterTraining = datetime.now()
    timeDelta = timeAfterTraining - timeBeforeTraining

    print("Tuning completed!")
    print("It took the network this time to train:", timeDelta)

    # Testing the model
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    # Calculating the accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

