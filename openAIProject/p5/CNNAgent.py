# Team Members: Yuzhang Guo, Zigeng Zhu

import numpy as np
import tensorflow as tf


class CNNAgent(object):
    def __init__(self, observation_space_dim, action_space,
                 learning_rate=0.1,
                 discount=0.99,
                 exploration_rate= 0.5,
                 exploration_decay_rate= 0.99,
                 batch_size=10):

        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay_rate
        self.discount = discount

        self.input_size = observation_space_dim
        self.output_size = action_space.n
        self._batch_size = batch_size

        self._sess = tf.Session()
        self._state = tf.placeholder(shape= [None, 210, 160, 3], dtype=tf.float32)
        self._cnnlayer1 = tf.contrib.layers.conv2d(num_outputs = 32, kernel_size  = (15,  9), inputs = self._state)
        self._cnnlayer2 = tf.contrib.layers.conv2d(inputs = self._cnnlayer1, kernel_size = (8, 6), num_outputs = 32)
        self._flatten = tf.contrib.layers.flatten(inputs = self._cnnlayer2)
        hidden_dim1 = 32
        hidden_dim2 = 16
        self._dense1 = tf.contrib.layers.fully_connected(inputs = self._flatten, num_outputs = 32, activation_fn = tf.nn.tanh)
        self.action_prob = tf.contrib.layers.fully_connected(inputs = self._dense1, num_outputs = action_space.n, activation_fn = tf.nn.softmax)

        log_prob = tf.log(self.action_prob + 1e-10)

        self._actions = tf.placeholder(tf.int32)
        self._discountedRewards = tf.placeholder(tf.float32)

        indices = tf.range(tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self._actions
        probs = tf.gather(tf.reshape(log_prob, [-1]), indices)

        self._loss = -tf.reduce_sum(tf.multiply(probs, self._discountedRewards))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self._train = optimizer.minimize(self._loss)

        self._sess.run([tf.global_variables_initializer()])

        self.saver = tf.train.Saver()

    def save(self, filename):
        self.saver.save(self._sess, filename)
        print("Model saved in path: %s" % filename)

    def load(self, filename):
        self.saver.restore(self._sess, filename)
        print("Model restored from path: %s" % filename)

    def reset(self):
        self.exploration_rate *= self.exploration_decay

    def act(self, observation):
        if np.random.random_sample() < self.exploration_rate:
            return np.random.randint(0, self.output_size)
        else:
            res = self._sess.run([self.action_prob], feed_dict={self._state: np.array([observation])})
            # print("res: " + str(res))
            return np.argmax(res)

    def update(self, observations, actions, rewards):
        dict = {self._state: observations, self._actions: actions,
                self._discountedRewards: self.discount_rewards(rewards)}
        self._sess.run(self._train, feed_dict=dict)

    def discount_rewards(self, rewards):
        disc_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount + rewards[t]
            disc_rewards[t] = running_add

        return disc_rewards
