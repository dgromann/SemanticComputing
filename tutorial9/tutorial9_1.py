# TU Dresden, SS 2018, Dagmar Gromann
# Regression implementation of a classifier for the MNIST dataset

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

# Setting Hyperparameters
learning_rate = 0.01
num_steps = 800
batch_size = 100
display_step = 100
epochs_between_evals = 10

# Network Parameters for a Network with 2 Hidden Layers
n_hidden_1 =  256 # Number of neurons of first layer
num_input = 784 # MNIST data input (image schape: 28 * 28)
num_classes = 10 # Total number of target classes in MNIST

# Loading Dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Graph input variables
#TODO: Create the target variable Y similar to X but with the correct shape
X = tf.placeholder(tf.float32, shape=[None, num_input]) # None means it can be a dimension of any size (to make it application to training and validation set); could also be batch_size


# Generate weights and biases for the layers
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, num_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create the model
def neural_net(x):
    #Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    #TODO: add a second layer - keep in mind that the connections between layer 1 and the output layer also need to be changed

    out_layer = tf.matmul(layer_1, weights['out'] + biases['out'])
    return out_layer

# Construct the model
logits = neural_net(X) # vector of raw non-normalized predictions that this classification model generates

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
#TODO: choose and specify an optimizer (tf.train...) with a predefined learnign rate

train_op = optimizer.minimize(loss_op)

# Evaluation of the model
prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

#Initialize the variables (that is, assign their default value)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})

            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

            # Calculate accuracy for MNIST validation image
            print("Validation Set Accuracy:",
                  sess.run(accuracy, feed_dict={X: mnist.validation.images,
                                                Y: mnist.validation.labels}))
    print("Optimization Finished!")

    # Evaluate the accuracy
    print("Test Set Accuracy:",
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                        Y: mnist.test.labels}))