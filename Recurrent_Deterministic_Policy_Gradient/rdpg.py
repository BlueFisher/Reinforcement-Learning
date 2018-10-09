import numpy as np
import tensorflow as tf

initializer_helper = {
    'kernel_initializer': tf.random_normal_initializer(0., 3),
    'bias_initializer': tf.constant_initializer(0.1)
}


class Actor(object):
    def __init__(self, sess, s_dim, a_dim, a_bound, lr, tau):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_bound = a_bound
        self.lr = lr

        with tf.variable_scope('actor'):
            self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')
            self.s = tf.placeholder(tf.float32, shape=(None, None, s_dim), name='state')
            self.s_ = tf.placeholder(tf.float32, shape=(None, None, s_dim), name='state_')

            self.initial_state = {}

            self.a, self.state = self._build_net(self.s, 'eval', True)
            self.a_, self.state_ = self._build_net(self.s_, 'target', False)

        self.param_eval = tf.global_variables('actor/eval')
        self.param_target = tf.global_variables('actor/target')

        # soft update
        self.target_replace_op = [tf.assign(t, tau * e + (1 - tau) * t)
                                  for t, e in zip(self.param_target, self.param_eval)]

    # s: (batch_size, time_steps, s_dim)
    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(64)
            self.initial_state[scope] = rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
            # (batch_size, time_steps, a_dim)
            l, state = tf.nn.dynamic_rnn(rnn_cell, inputs=s, initial_state=self.initial_state[scope], dtype=tf.float32)
            l = tf.reshape(l, [-1, l.shape[2]])
            l = tf.layers.dense(l, 128, activation=tf.nn.relu, **initializer_helper)
            a = tf.layers.dense(l, self.a_dim, activation=tf.tanh, **initializer_helper)
            a = tf.reshape(a, [self.batch_size, -1, a.shape[1]])
            a = a * self.a_bound
        return a, state

    # s: (s_dim, )
    def choose_action(self, s, initial_state=False):
        # s: (1, 1, s_dim)
        s = s[np.newaxis, np.newaxis, :]
        # a: (1, 1, a_dim)
        if initial_state:
            a, self.temp_state = self.sess.run([self.a, self.state], {
                self.s: s,
                self.batch_size: 1,
            })
        else:
            a, self.temp_state = self.sess.run([self.a, self.state], {
                self.s: s,
                self.batch_size: 1,
                self.initial_state['eval']: self.temp_state
            })

        return a.reshape((-1,))

    def generate_gradients(self, Q_a_gradients):
        # 根据链式法则，生成 Actor 的梯度
        grads = tf.gradients(self.a, self.param_eval, Q_a_gradients)
        with tf.variable_scope('actor/train'):
            optimizer = tf.train.AdamOptimizer(-self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, self.param_eval))

    # s: (batch_size, None, s_dim)
    def learn(self, s):
        self.sess.run(self.train_op, {
            self.s: s,
            self.batch_size: s.shape[0]
        })
        self.sess.run(self.target_replace_op)


class Critic(object):
    def __init__(self, sess, s_dim, a_dim, s, a, s_, a_, batch_size, gamma, lr, tau):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.s = s
        self.a = a
        self.s_ = s_
        self.batch_size=batch_size

        with tf.variable_scope('critic'):
            # (batch_size, num_steps, 1)
            self.r = tf.placeholder(tf.float32, shape=(None, None, 1), name='rewards')
            # self.a = tf.placeholder(tf.float32, shape=(None, None, a_dim), name='action')
            # (batch_size, num_steps, 1)
            self.q, _ = self._build_net(s, a, 'eval', True)
            q_, _ = self._build_net(s_, a_, 'target', False)

            param_eval = tf.global_variables('critic/eval')
            param_target = tf.global_variables('critic/target')
            # soft update
            self.target_replace_op = [tf.assign(t, tau * e + (1 - tau) * t)
                                      for t, e in zip(param_target, param_eval)]

            # y_t
            y = self.r + gamma * q_
            # y = tf.stop_gradient(y)

            self.loss = loss = tf.reduce_mean(tf.squared_difference(y, self.q))
            with tf.variable_scope('train'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(loss, var_list=param_eval)

    # s:(batch_size, num_steps, s_dim)
    # a:(batch_size, num_steps, a_dim)
    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            # (batch_size, num_steps, s_dim+a_dim)
            # h = tf.concat([s, a], 2)
            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(64)
            initial_state = rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
            # (batch_size, num_steps, 1)
            sl, state = tf.nn.dynamic_rnn(rnn_cell, inputs=s, initial_state=initial_state, dtype=tf.float32)
            sl = tf.reshape(sl, [-1, sl.shape[2]])
            al = tf.reshape(a, [-1, a.shape[2]])
            al = tf.layers.dense(al, 64, activation=None, **initializer_helper)
            l = sl + al
            l = tf.layers.dense(l, 128, activation=tf.nn.relu, **initializer_helper)

            q = tf.layers.dense(l, 1, **initializer_helper)
            q = tf.reshape(q, [self.batch_size, -1, 1])

        return q, state

    # 生成 Q 对 a 的导数，交给 actor
    def get_gradients(self):
        grad = tf.gradients(self.q, self.a)
        return grad[0]

    # s: (batch_size, num_steps, s_dim)
    # a: (batch_size, num_steps, a_dim)
    # r: (batch_size, num_steps, 1)
    # s_: (batch_size, num_steps, s_dim)
    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, {
            self.s: s,
            self.a: a,
            self.r: r,
            self.s_: s_,
            self.batch_size: s.shape[0]
        })

    def replace(self):
        self.sess.run(self.target_replace_op)
