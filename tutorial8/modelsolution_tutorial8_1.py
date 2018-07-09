import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Exercise 1a: basic operations in tensorflow
# TODO: before you run this code, what do you think is the output of the three print statements?
x = 45
y = x + 5
print("Exercise 1a: x + 5 if x is 45 is ", y)

x = tf.constant(50, name='x')
y = tf.Variable(x+5, name='y')
print("Exercise 1a: Do we also get 50 in tensorflow? ", y)

model1 = tf.global_variables_initializer()
with tf.Session() as session1:
    session1.run(model1)
    print("Exercise 1a: What about now? ", session1.run(y))


# # Exercise 1b: What about arrays?
# # TODO: try to predict what this code will do
# a = tf.constant([35, 40, 45], name='a')
# b = tf.Variable(a + 5, name='b')
#
#
# model2 = tf.global_variables_initializer()
#
# with tf.Session() as session2:
#     session2.run(model2)
#     print("Exercise 1b: And for an array ", session2.run(b))

# # Exercise 1c: We can also assign values dynamically
# # TODO: try to predict what this code will do
# x = tf.Variable(0, name='x')
#
# model = tf.global_variables_initializer()
#
# with tf.Session() as session:
#     session.run(model)
#     for i in range(5):
#         x = x + 1
#         print("Exercise 1c: Dynmamic assignment ", session.run(x))

# # Exercise 1d: Understanding placeholders
# # TODO: try to predict what this code will do
# # Placeholders are used for training data which is fed into the code when running the model
# a = tf.placeholder("float")
# b = tf.placeholder("float")
# y = tf.multiply(a, b)
#
# feed_dict ={a: 2, b: 3}
#
# with tf.Session() as sess:
#     print(sess.run(y, feed_dict))

#Exercise 1e: Use the above exercises to fill in the code template below
#TODO: Calculate the max value (tf.argmax) and the average (tf.reduce_mean) of the random numpy array (data)
z = tf.Variable(0.0, name='z')

with tf.Session() as session:
    for i in range(5):
        data = np.random.randint(1000, size=10000)
        z = data
        model = tf.global_variables_initializer()
        session.run(model)
        print(session.run(tf.argmax(z)))
        print(session.run(tf.reduce_mean(z)))

#TODO: Take a look at what the graph represents
# graph = tf.get_default_graph()
# for op in graph.get_operations():
#     print(op.name)


#Exercise 1f: Let's train a full linear regression in tensorflow
# trainX has values between -1 and 1, and trainY has 3 times the trainX and some randomness
trainX = np.linspace(-1, 1, 101)
trainY = 3 * trainX + np.random.randn(*trainX.shape) * 0.33

#TODO: Create two float placeholders to be used in training later called "X" and "Y"
X = tf.placeholder("float")
Y = tf.placeholder("float")

#TODO: Create a weights variable called "w" as a float (0.0) with the name "weights"
#TODO: Create a variable called y_model that multiplies our training data (X) with the weights (w)
w = tf.Variable(0.0, name="weights")
y_model = tf.multiply(X, w)

#TODO: What kind of cost function is this? What does the number with the GradientDescentOptimizer mean?
cost = (tf.pow(Y-y_model, 2))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        for (x, y) in zip(trainX, trainY):
            sess.run(train_op, feed_dict={X: x, Y: y})
    print(sess.run(w))
