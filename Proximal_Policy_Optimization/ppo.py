import numpy as np
import tensorflow as tf


class PPO(object):
    def __init__(self, s_dim, a_bound, c1, c2, epsilon, lr, K):
        self.sess = tf.Session()

        self.a_bound = a_bound
        self.K = K

        self.s = tf.placeholder(tf.float32, shape=(None, s_dim), name='s_t')

        pi, self.v, params = self._build_net('network', True)
        old_pi, old_v, old_params = self._build_net('old_network', False)

        self.discounted_r = tf.placeholder(tf.float32, shape=(None, 1), name='discounted_r')

        advantage = self.discounted_r - old_v

        self.a = tf.placeholder(tf.float32, shape=(None, 1), name='a_t')
        ratio = pi.prob(self.a) / old_pi.prob(self.a)

        L_clip = tf.reduce_mean(tf.minimum(
            ratio * advantage,  # 替代的目标函数 surrogate objective
            tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * advantage
        ))
        L_vf = tf.reduce_mean(tf.square(self.discounted_r - self.v))
        S = tf.reduce_mean(pi.entropy())
        L = L_clip - c1 * L_vf + c2 * S

        self.choose_action_op = tf.squeeze(pi.sample(1), axis=0)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(-L)
        self.update_params_op = [tf.assign(r, v) for r, v in zip(old_params, params)]

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self, scope, trainable):
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(self.s, 100, tf.nn.relu, trainable=trainable)
            mu = tf.layers.dense(l1, 1, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, 1, tf.nn.softplus, trainable=trainable)

            # 状态价值函数 v 与策略 π 共享同一套神经网络参数
            v = tf.layers.dense(l1, 1, trainable=trainable)

            mu, sigma = mu * self.a_bound, sigma + 1

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

        params = tf.global_variables(scope)
        return norm_dist, v, params

    def get_v(self, s):
        return self.sess.run(self.v, {
            self.s: np.array([s])
        }).squeeze()

    def choose_action(self, s):
        a = self.sess.run(self.choose_action_op, {
            self.s: s[np.newaxis, :]
        })[0]
        return np.clip(a, -self.a_bound, self.a_bound)

    def train(self, s, a, discounted_r):
        self.sess.run(self.update_params_op)

        # K epochs
        for i in range(self.K):
            self.sess.run(self.train_op, {
                self.s: s,
                self.a: a,
                self.discounted_r: discounted_r
            })
