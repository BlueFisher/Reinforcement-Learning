import numpy as np
import tensorflow as tf
import gym


initializer_helper = {
    'kernel_initializer': tf.random_normal_initializer(0., 0.1),
    'bias_initializer': tf.constant_initializer(0.1)
}


class Actor(object):
    def __init__(self, sess, s_dim, a_bound, lr):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, shape=(1, s_dim), name='s')
        self.a = tf.placeholder(tf.float32, shape=(), name='a')
        self.td_error = tf.placeholder(tf.float32, shape=(), name='td_error')

        l1 = tf.layers.dense(inputs=self.s, units=30, activation=tf.nn.relu, **initializer_helper)

        # 均值
        mu = tf.layers.dense(inputs=l1, units=1, activation=tf.nn.tanh, **initializer_helper)
        # 方差
        sigma = tf.layers.dense(inputs=l1, units=1, activation=tf.nn.softplus, **initializer_helper)

        # 均值控制在(-2, 2) 方差控制在(0, 2)
        mu, sigma = tf.squeeze(mu * a_bound), tf.squeeze(sigma + 1)

        self.normal_dist = tf.distributions.Normal(mu, sigma)
        self.action = tf.clip_by_value(self.normal_dist.sample(1), -a_bound, a_bound)

        loss = self.normal_dist.log_prob(self.a) * self.td_error

        # 最大化 J，即最小化 -loss
        self.optimizer = tf.train.AdamOptimizer(lr).minimize(-loss)

    def learn(self, s, a, td_error):
        self.sess.run(self.optimizer, feed_dict={
            self.s: s[np.newaxis, :],
            self.a: a,
            self.td_error: td_error
        })

    def choose_action(self, s):
        return self.sess.run(self.action, {
            self.s: s[np.newaxis, :]
        }).squeeze()


# 与 actor_critic_cartpole.py 中 Critic 相同
class Critic(object):
    def __init__(self, sess, s_dim, gamma, lr):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, shape=(1, s_dim), name='s')
        self.r = tf.placeholder(tf.float32, shape=None, name='r')
        self.v_ = tf.placeholder(tf.float32, shape=None, name='v_')

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


env = gym.make('Pendulum-v0')
env = env.unwrapped

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

with tf.Session() as sess:
    actor = Actor(
        sess,
        state_dim,
        action_bound,
        0.001
    )
    critic = Critic(
        sess,
        state_dim,
        0.9,
        0.001
    )

    tf.global_variables_initializer().run()

    for i_episode in range(1000):
        s = env.reset()

        ep_reward = 0
        for j in range(200):
            # env.render()
            a = actor.choose_action(s)
            s_, r, done, _ = env.step([a])
            r /= 10
            ep_reward += r

            td_error = critic.learn(s, r, s_)
            actor.learn(s, a, td_error)
            if j == 200 - 1:
                print(i_episode, int(ep_reward))
                break

            s = s_
