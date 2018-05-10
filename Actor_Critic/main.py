import numpy as np
import tensorflow as tf
import random
import gym


initializer_helper = {
    'kernel_initializer': tf.random_normal_initializer(0., 0.1),
    'bias_initializer': tf.constant_initializer(0.1)
}


class Actor(object):
    def __init__(self, sess, s_dim, a_dim, lr):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.s = tf.placeholder(tf.float32, shape=(1, s_dim), name='s')
        self.a = tf.placeholder(tf.float32, shape=(a_dim,), name='a')
        self.td_error = tf.placeholder(tf.float32, shape=(), name='td_error')

        l = tf.layers.dense(
            self.s, a_dim, **initializer_helper
        )

        self.a_prob_z = tf.nn.softmax(l)  # 每个行为所对应的概率

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.squeeze(l), labels=self.a) * self.td_error
        # 与以下形式相同，这里用softmax交叉熵代替
        # loss = tf.reduce_sum(-tf.log(self.a_prob_z) * self.a) * self.td_error
        self.optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    def choose_action(self, s):  # 根据softmax所输出的概率选择行为
        a_prob_z = self.sess.run(self.a_prob_z, feed_dict={
            self.s: s[np.newaxis, :]
        })

        action = np.random.choice(range(self.a_dim), p=a_prob_z.ravel())
        return action

    def learn(self, s, a, td_error):
        one_hot_action = np.zeros(self.a_dim)
        one_hot_action[a] = 1
        self.sess.run(self.optimizer, feed_dict={
            self.s: s[np.newaxis, :],
            self.a: one_hot_action,
            self.td_error: td_error
        })


class Critic(object):
    def __init__(self, sess, s_dim, gamma, lr):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, shape=(1, s_dim), name='s')
        self.v_ = tf.placeholder(tf.float32, shape=(), name='v_')
        self.r = tf.placeholder(tf.float32, shape=(), name='r')

        l = tf.layers.dense(
            inputs=self.s, units=30,
            activation=tf.nn.relu, **initializer_helper
        )
        self.v = tf.layers.dense(
            inputs=l, units=1, **initializer_helper
        )

        self.td_error = self.r + gamma * self.v_ - self.v
        loss = tf.square(self.td_error)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    def learn(self, s, r, s_):
        v_ = self.sess.run(self.v, {
            self.s: s_[np.newaxis, :]
        })

        td_error, _ = self.sess.run([self.td_error, self.train_op], {
            self.s: s[np.newaxis, :],
            self.v_: v_.squeeze(),
            self.r: r
        })

        return td_error.squeeze()


env = gym.make('CartPole-v0')
env = env.unwrapped

with tf.Session() as sess:
    actor = Actor(
        sess=sess,
        s_dim=4,
        a_dim=2,
        lr=0.01
    )

    critic = Critic(
        sess=sess,
        s_dim=4,
        gamma=0.99,
        lr=0.001
    )

    tf.global_variables_initializer().run()

    for i_episode in range(10000):
        s = env.reset()
        n_step = 0
        while True:
            a = actor.choose_action(s)
            s_, r, done, _ = env.step(a)

            if done:
                r = -20

            td_error = critic.learn(s, r, s_)
            actor.learn(s, a, td_error)

            n_step += 1
            if done:
                print(i_episode, n_step)
                break

            s = s_
