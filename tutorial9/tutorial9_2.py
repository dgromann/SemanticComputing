# TU Dresden, SS 2018, Dagmar Gromann
# LSTM/GRU implementation of a classifier for the MNIST dataset

import os
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

# Setting Hyperparameters
learning_rate = 0.01
num_steps = 400
batch_size = 100
display_step = 100
epochs_between_evals = 10

# Network Parameters for a Network with 2 Hidden Layers
time_steps = 28 # undolled through 28 time steps
n_hidden_1 =  256 # Number of neurons of first layer
num_input = 28 # rows of 28 pixels
num_classes = 10 # Total number of target classes in MNIST

# Loading Dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Graph input variables
X = tf.placeholder(tf.float32, shape=[None, time_steps, num_input]) # None means it can be a dimension of any size (to make it application to training and validation set); could also be batch_size
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

# Generate weights and biases for the layers
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden_1, num_classes]))
}

biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create the model
def neural_net(x):
    # Processing the input tensor from [batch_size, time_steps, num_input] to [batch_size, n_classes] by out_weight multiplication
    input = tf.unstack(x, time_steps, 1)

    # Defining the network
    # TODO: experiment with forget bias and change the GRU to a BasicLSTMCell
    lstm_layer = rnn.LSTMCell(n_hidden_1, name="layer1")
    outputs,_= rnn.static_rnn(lstm_layer, input, dtype="float")

    # Converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
    out_layer = tf.matmul(outputs[-1], weights['out'] + biases['out'])
    return out_layer

# Construct the model
logits = neural_net(X) # vector of raw non-normalized predictions that this classification model generates

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluation of the model
prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

#Initialize the variables (that is, assign their default value)
init = tf.global_variables_initializer()

# Save the model (only the last four ones are kept)
saver = tf.train.Saver(max_to_keep=4)

with tf.Session() as sess:
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x=batch_x.reshape((batch_size,time_steps,num_input))

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})

            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

            val_data = mnist.validation.images[:128].reshape((-1, time_steps, num_input))
            val_label = mnist.validation.labels[:128]
            # Calculate accuracy for MNIST validation image
            print("Validation Set Accuracy:",
                  sess.run(accuracy, feed_dict={X: val_data,
                                                Y: val_label}))

    print("Optimization Finished!")
    saver.save(sess, './model')

    test_data = mnist.test.images[:128].reshape((-1, time_steps, num_input))
    test_label = mnist.test.labels[:128]
    # Evaluate the accuracy
    print("Test Set Accuracy:",
        sess.run(accuracy, feed_dict={X: test_data,
                                        Y: test_label}))

## If we want to load the previously stored model again
#with tf.Session() as sess:
#    new_saver = tf.train.import_meta_graph('model.meta')
#    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
