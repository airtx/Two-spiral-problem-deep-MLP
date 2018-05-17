# -*- coding: utf-8 -*-
"""
Created on Fri May  4 14:45:13 2018

@author: rafael

Lista 02 - Q05 (a)

Objetivo: Tentativa de resolver o problema das duas espirais utilizando uma rede Deep
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

## import data
dados = np.load('data.npz')

## extract training data
train_data = dados['train_data']
train_labels = dados['train_labels']

## extract validation data
test_data = dados['test_data']
test_labels = dados['test_labels']

## free memory
del dados

## size of data
num_examples = train_data.shape[0]
num_test = test_data.shape[0]

## shuffl data
s = np.arange(num_examples)
np.random.shuffle(s)
train_data = train_data[s]
train_labels = train_labels[s]

## training parameters
learning_rate = 0.002
training_epochs = 50000
batch_size = 5
display_step = 10

## regularization
## This is a good beta value to start with
beta = 0.013

## network parameters
fl = 30 # parameters to change neurons easily
n_hidden_1 = fl # number of neurons in first layer
n_hidden_2 = int(fl*2/3) # number of neurons in second layer
n_hidden_3 = int(fl*2/6) # number of neurons in third layer
n_hidden_4 = int(fl*2/12) # number of neurons in fourth layer
n_hidden_5 = int(fl*2/24) # number of neurons in fifth layer
n_input = 2 # number of inputs
n_classes = 1 # number of classes in output

## input and output tensors
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

## model criation
def multilayer_perceptron(x, weights, biases):
    ## hidden 1
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.leaky_relu(layer_1)
    ## hidden 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.leaky_relu(layer_2)
    ## hidden 3
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.leaky_relu(layer_3)
    ## hidden 4
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.leaky_relu(layer_4)
    ## hidden 5
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.leaky_relu(layer_5)
    ## output
    out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
    out_layer = tf.tanh(out_layer)
    return out_layer

## variables for weights and bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
    'out': tf.Variable(tf.random_normal([n_hidden_5, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'b3': tf.Variable(tf.zeros([n_hidden_3])),
    'b4': tf.Variable(tf.zeros([n_hidden_4])),
    'b5': tf.Variable(tf.zeros([n_hidden_5])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

## model training 
pred = multilayer_perceptron(x, weights, biases)

## cost function
cost = tf.reduce_mean(tf.losses.mean_squared_error(labels = y, predictions = pred))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

## loss function and L2 regularization
regularizer = tf.nn.l2_loss(weights['h1'])+tf.nn.l2_loss(weights['h2'])+tf.nn.l2_loss(weights['h3'])+tf.nn.l2_loss(weights['h4'])+tf.nn.l2_loss(weights['h5'])
loss = tf.reduce_mean(cost + beta * regularizer)
## optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

## variables initialization
init = tf.global_variables_initializer()

## initial cost
avg_cost = 2

## session start
with tf.Session() as sess:
    sess.run(init)
    print('Start training')
    ## training loop
    for epoch in range(training_epochs):
        if(avg_cost < 0.05):
            break #break if cost is small enough
        avg_cost = 0.
        total_batch = int(num_examples/batch_size)
        ## batch loop
        for i in range(total_batch):
            batch_x, batch_y = [train_data[i*batch_size:(i + 1)*batch_size], train_labels[i*batch_size:(i + 1)*batch_size]]
            ## run backprop and loss function
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # average cost value
            avg_cost += c / total_batch
        ## display results
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.6f}".format(avg_cost))
    print("Finish training!")

    # Test model
    correct_prediction = tf.equal(tf.sign(pred), y)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), axis = 0)
    print("Accuracy:", accuracy.eval({x: test_data, y: test_labels}))
    ############################################################################
    ## graph plot
    fig = plt.figure()
    # create a mesh to plot in
    x_min, x_max = [-5,5]
    y_min, y_max = [-5,5]
    h = abs((x_max / x_min)/100)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h, dtype = 'float32'),
    np.arange(y_min, y_max, h, dtype = 'float32'))
    
    ## calculate predictions
    Z = np.sign(pred.eval({x : np.c_[xx.ravel(), yy.ravel()]}))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    
    plt.scatter(train_data[:,0], train_data[:,1], c=train_labels.reshape(num_examples), cmap=plt.cm.Paired, edgecolors ='black')
    plt.scatter(test_data[:,0], test_data[:,1], c=test_labels.reshape(num_test), cmap=plt.cm.Paired, edgecolors ='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(xx.min(), xx.max())
    plt.title('MLP (Deep)')
    plt.axis([-5,5,-5,5])
    
    ## calculate confusion matrix
    CM = confusion_matrix(test_labels, np.sign(pred.eval({x : test_data})))
    print("Matriz de confusão: ")
    print(CM)

plt.show()