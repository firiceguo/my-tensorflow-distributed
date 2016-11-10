import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")

import Utils


class Autoencoder(object):

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.sub(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        self.init_op = tf.initialize_all_variables()
        # self.sess = tf.Session()
        # self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        # all_weights['w1'] = tf.Variable(autoencoder.Utils.xavier_init(self.n_input, self.n_hidden))
        all_weights['w1'] = tf.Variable(Utils.xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X, session):
        try:
            cost, opt = session.run((self.cost, self.optimizer), feed_dict={self.x: X})
        except:
            cost = 0
        return cost

    def calc_total_cost(self, X, session):
        return session.run(self.cost, feed_dict={self.x: X})

    def transform(self, X, session):
        return session.run(self.hidden, feed_dict={self.x: X})

    def generate(self, session, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return session.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X, session):
        return session.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self, session):
        return session.run(self.weights['w1'])

    def getBiases(self, session):
        return session.run(self.weights['b1'])
