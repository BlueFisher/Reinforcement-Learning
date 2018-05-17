import numpy as np
import tensorflow as tf

initializer_helper = {
    'kernel_initializer': tf.random_normal_initializer(0., 0.1),
    'bias_initializer': tf.constant_initializer(0.1)
}


class Actor(object):
    def __init__(self, sess, s_dim, a_bound, lr, tau):
        self.sess = sess 
        self.s_dim = s_dim
        self.a_bound = a_bound
        self.lr = lr

        with tf.variable_scope('actor'):
            self.s = tf.placeholder(tf.float32, shape=(None, s_dim), name='state')
            self.s_ = tf.placeholder(tf.float32, shape=(None, s_dim), name='state_')

            self.a = self._build_net(self.s, 'eval', True)
            self.a_ = self._build_net(self.s_, 'target', False)

        self.param_eval = tf.global_variables('actor/eval')
        self.param_target = tf.global_variables('actor/target')

        # soft update
        self.target_replace_ops = [tf.assign(t, tau * e + (1 - tau) * t) for t, e in zip(self.param_target, self.param_eval)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            l = tf.layers.dense(
                s, 30, activation=tf.nn.relu,
                name='layer', trainable=trainable, **initializer_helper
            )
            with tf.variable_scope('action'):
                a = tf.layers.dense(
                    l, 1, activation=tf.nn.tanh,
                    name='action', trainable=trainable, **initializer_helper
                )
                a = a * self.a_bound
        return a

    def choose_action(self, s):
        a = self.sess.run(self.a, {
            self.s: s[np.newaxis, :]
        })

        return a[0]

    def generate_gradients(self, Q_a_gradients):
        # 根据链式法则，生成 Actor 的梯度
        grads = tf.gradients(self.a, self.param_eval, Q_a_gradients)
        optimizer = tf.train.AdamOptimizer(-self.lr)
        self.train_ops = optimizer.apply_gradients(zip(grads, self.param_eval))

    def learn(self, s):
        self.sess.run(self.train_ops, {
            self.s: s
        })
        self.sess.run(self.target_replace_ops)


class Critic(object):
    def __init__(self, sess, s_dim, s, s_, a, a_, gamma, lr, tau):
        self.sess = sess
        self.s_dim = s_dim
        self.s = s
        self.s_ = s_
        self.a = a

        with tf.variable_scope('critic'):
            self.r = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
            self.q = self._build_net(s, a, 'eval', True)
            self.q_ = self._build_net(s_, a_, 'target', False)

        param_eval = tf.global_variables('critic/eval')
        param_target = tf.global_variables('critic/target')
        # soft update
        self.target_replace_ops = [tf.assign(t, tau * e + (1 - tau) * t)
                                   for t, e in zip(param_target, param_eval)]

        # y_t
        target_q = self.r + gamma * self.q_
        # 可以保留或忽略 target_q 的梯度
        target_q = tf.stop_gradient(target_q)

        loss = tf.reduce_mean(tf.squared_difference(target_q, self.q))
        self.train_ops = tf.train.AdamOptimizer(lr).minimize(loss, var_list=param_eval)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            # ls = tf.layers.dense(
            #     s, 30, name='layer_s', trainable=trainable, **initializer_helper
            # )
            # la = tf.layers.dense(
            #     a, 30, name='layer_a', trainable=trainable, **initializer_helper
            # )
            # l = tf.nn.relu(ls + la)

            l = tf.concat([s, a], 1)
            l = tf.layers.dense(l, 30, activation=tf.nn.relu, trainable=trainable, **initializer_helper)

            with tf.variable_scope('Q'):
                q = tf.layers.dense(l, 1, name='q', trainable=trainable, **initializer_helper)
        return q

    # 生成 Q 对 a 的导数，交给 actor
    def get_gradients(self):
        return tf.gradients(self.q, self.a)[0]

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_ops, {
            self.s: s,
            self.a: a,
            self.r: r,
            self.s_: s_,
        })
        self.sess.run(self.target_replace_ops)
