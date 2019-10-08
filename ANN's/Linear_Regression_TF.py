import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

np.random.seed(101) 
tf.set_random_seed(101) 

x = np.linspace(100.0, 0.0, 50)
y = np.linspace(100.0, 0.0, 50)

x += np.random.uniform(-4.5, 4.5, 50) 
y += np.random.uniform(-4.5, 4.5, 50) 

n = len(x)

y_test = np.random.randn(len(y))

X = tf.placeholder("float")
Y = tf.placeholder("float")
W = tf.Variable(np.random.randn(), name="W")
b = tf.Variable(np.random.randn(), name="b")

learning_rate = 0.001
training_iteration = 800

y_pred = tf.add(tf.multiply(X, W), b)

cost_function = tf.reduce_sum(tf.pow(y_pred - Y, 2))/(2 * n)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    display_step = 20
    for iteration in range(training_iteration):
        for (_x, _y) in zip(x, y):
            sess.run(optimizer, feed_dict={X:_x, Y:_y})

        if iteration % display_step == 0:
            c = sess.run(cost_function, feed_dict={X:x, Y:y})
            print("Iteration:", iteration, "cost=", c, "W=", sess.run(W), "b=", sess.run(b))
    
    training_cost = sess.run(cost_function, feed_dict ={X: x, Y: y}) 
    weight = sess.run(W) 
    bias = sess.run(b)

    predictions = weight * x + bias 
    print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n')

    plt.figure()
    plt.plot(x, y, 'ro', label ='Original data')
    plt.plot(x, predictions, label ='Fitted line') 
    plt.legend()
    plt.show()
    
    print('MAE:', metrics.mean_absolute_error(y_test,predictions))
    print('MSE:', metrics.mean_squared_error(y_test,predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,predictions)))