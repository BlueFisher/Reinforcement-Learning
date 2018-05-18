import tensorflow as tf
import concurrent.futures
import numpy as np
import gym
import threading


class Global_net(object):
    def __init__(self, sess, s_dim, a_bound, gamma, actor_lr, critic_lr, max_global_ep, max_ep_steps, update_iter):
        self.sess = sess
        self.a_bound = a_bound
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.global_ep = 0
        self.max_global_ep = max_global_ep
        self.max_ep_steps = max_ep_steps
        self.update_iter = update_iter

        self.s = tf.placeholder(tf.float32, shape=(None, s_dim), name='s')

        *_, self.a_params, self.c_params = self.build_net('global')

    def build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope(scope):
            with tf.variable_scope('actor'):
                l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
                mu = tf.layers.dense(l_a, 1, tf.nn.tanh, kernel_initializer=w_init, name='mu')
                sigma = tf.layers.dense(l_a, 1, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
            with tf.variable_scope('critic'):
                l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
                v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')

            mu, sigma, v = tf.squeeze(mu * self.a_bound), tf.squeeze(sigma + 1e-4), tf.squeeze(v)

            a_params = tf.global_variables(scope + '/actor')
            c_params = tf.global_variables(scope + '/critic')
            return mu, sigma, v, a_params, c_params


# 独立线程，actor critic 合并
class Worker_net(object):
    def __init__(self, global_net, name):
        self.g = global_net
        self.name = name

        self.a = tf.placeholder(tf.float32, shape=(None,), name='a')
        self.R = tf.placeholder(tf.float32, shape=(None,), name='R')

        mu, sigma, self.v, self.a_params, self.c_params = global_net.build_net(name)

        td = self.R - self.v
        critic_loss = tf.reduce_mean(tf.square(td))

        self.normal_dist = tf.distributions.Normal(mu, sigma)
        obj = self.normal_dist.log_prob(self.a) * tf.stop_gradient(td)
        # 加上策略的熵增加探索空间，避免过早进入局部最优
        obj = obj + self.normal_dist.entropy()
        actor_loss = -tf.reduce_mean(self.normal_dist.log_prob(self.a) * tf.stop_gradient(td))

        self._choose_a_ops = tf.squeeze(tf.clip_by_value(self.normal_dist.sample(1),
                                                         -global_net.a_bound, global_net.a_bound))

        self.a_grads = tf.gradients(actor_loss, self.a_params)
        self.c_grads = tf.gradients(critic_loss, self.c_params)

        # 用自己的梯度来更新全局参数
        self.update_a_op = tf.train.RMSPropOptimizer(self.g.actor_lr).apply_gradients(zip(self.a_grads, self.g.a_params))
        self.update_c_op = tf.train.RMSPropOptimizer(self.g.critic_lr).apply_gradients(zip(self.c_grads, self.g.c_params))

        self.sync_a_ops = [tf.assign(l, g) for l, g in zip(self.a_params, self.g.a_params)]
        self.sync_c_ops = [tf.assign(l, g) for l, g in zip(self.c_params, self.g.c_params)]

    def _choose_action(self, s):
        return self.g.sess.run(self._choose_a_ops, {
            self.g.s: s[np.newaxis, :]
        })

    # 从全局下载参数替换子线程中的参数
    def _sync(self):
        self.g.sess.run(self.sync_a_ops)
        self.g.sess.run(self.sync_c_ops)

    # 在子线程中进行学习，并将子线程的参数更新到全局
    def _update(self, done, transition):
        if done:
            R = 0
        else:
            s_ = transition[-1][2]
            R = self.g.sess.run(self.v, {
                self.g.s: s_[np.newaxis, :]
            }).squeeze()

        buffer_s, buffer_a, _, buffer_r = zip(*transition)
        buffer_R = []
        for r in buffer_r[::-1]:
            R = r + self.g.gamma * R
            buffer_R.append(R)

        buffer_R.reverse()

        buffer_s, buffer_a, buffer_R = np.vstack(buffer_s), np.array(buffer_a), np.array(buffer_R)

        self.g.sess.run([self.update_a_op, self.update_c_op], {
            self.g.s: buffer_s,
            self.a: buffer_a,
            self.R: buffer_R
        })

    # 子线程模拟自己独有的环境
    def run(self):
        env = gym.make('Pendulum-v0')
        env = env.unwrapped

        self._sync()
        total_step = 1
        transition = []

        while self.g.global_ep <= self.g.max_global_ep:
            s = env.reset()
            ep_rewards = 0
            for ep_i in range(self.g.max_ep_steps):
                a = self._choose_action(s)
                s_, r, *_ = env.step([a])
                done = ep_i == self.g.max_ep_steps - 1

                ep_rewards += r
                transition.append((s, a, s_, r / 10))

                if total_step % self.g.update_iter == 0 or done:
                    self._update(done, transition)
                    self._sync()
                    transition = []

                s = s_
                total_step += 1

            self.g.global_ep += 1
            print(self.g.global_ep, self.name, int(ep_rewards))


sess = tf.Session()


global_net = Global_net(
    sess=sess,
    s_dim=3,
    a_bound=2,
    gamma=0.9,
    actor_lr=0.0001,
    critic_lr=0.001,
    max_global_ep=1000,
    max_ep_steps=200,
    update_iter=10
)

THREAD_N = 4

workers = [Worker_net(global_net, 'w' + str(i)) for i in range(THREAD_N)]

sess.run(tf.global_variables_initializer())

executor = concurrent.futures.ThreadPoolExecutor(THREAD_N)
futures = [executor.submit(w.run) for w in workers]
concurrent.futures.wait(futures)
for f in futures:
    f.result()
