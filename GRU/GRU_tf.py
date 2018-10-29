import tensorflow as tf
import numpy as np
class GRU():

    def __init__(self):

        self.n_input = None
        self.n_output = None
        self.n_neurons = None
        self.n_steps = None
        self.n_layers = None

    def interface(self,X,n_inputs,n_outputs,n_neurons,n_steps,n_layers):
        '''
        This method provide network model using TensorFlow.
        You can customize using below parameters.
        :param n_inputs: Number of model input
        :param n_outputs: Number of model output
        :param n_neurons: Number of neuron
        :param n_steps: Number of step
        :param n_layers: Number of layer
        :return:
        y: output tensor
        '''
        print('Create Net work model')
        layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.leaky_relu)
                  for layer in range(n_layers)]
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
        rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
        outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
        y = outputs[:, n_steps - 1, :]

        return y

    def loss(self,y,t):
        '''
        This method calculate loss.
        :param y: Prediction data
        :param t: Target data
        :return:
        loss: Meas squard error
        '''
        print('calculate loss')
        with tf.name_scope("Loss"):
            loss = tf.reduce_mean(tf.square(y - t))  # loss function = mean squared error
        return loss


    def training(self,loss,learning_rate):
        '''
        This method try to minimize loss
        :param loss: loss calculate by prediction data and target data.
        :param learning_rate: Learning rate. Hyper parameter. If decrease this value
                            optimization will proceed baby steps.
        :return:
        Training_steps:
        '''
        print('Optimization model for training of machine learning algorithm')
        with tf.name_scope("Training"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            training_steps = optimizer.minimize(loss)
        return training_steps

