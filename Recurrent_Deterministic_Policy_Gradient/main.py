import gym
import numpy as np
import tensorflow as tf
import random

from rdpg import Actor, Critic
from memory import *

if __name__ == '__main__':

    env = gym.make('Pendulum-v0')
    env = env.unwrapped

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    var = 3.
    BATCH_SIZE = 10
    LEN_EPISODE = 200

    with tf.Session() as sess:
        memory = Memory(BATCH_SIZE, 64)
        actor = Actor(sess, state_dim, action_dim, action_bound,
                      lr=0.001, tau=0.01)
        critic = Critic(sess, state_dim, action_dim, actor.s, actor.a, actor.s_, actor.a_,
                        batch_size=actor.batch_size, gamma=0.9, lr=0.001, tau=0.01)
        actor.generate_gradients(critic.get_gradients())

        sess.run(tf.global_variables_initializer())

        for i in range(10000):
            s = env.reset()
            r_episode = 0
            b_s, b_a, b_r, b_s_ = [], [], [], []

            for j in range(200):
                a = actor.choose_action(s, initial_state=j == 0)

                a = np.clip(np.random.normal(a, var), -action_bound, action_bound)  # 异策略探索
                s_, r, *_ = env.step(a)
                b_s.append(s)
                b_a.append(a)
                b_r.append([(r + 8)])
                b_s_.append(s_)

                r_episode += r
                s = s_

            if memory.isFull:
                var *= 0.9995
            memory.store_transition(b_s, b_a, b_r, b_s_)
            print('episode {}\treward {:.2f}\tvar {:.2f}'.format(i, r_episode, var))

            if not memory.can_batch():
                continue

            b_s, b_a, b_r, b_s_ = memory.get_mini_batches()
            critic.learn(b_s, b_a, b_r, b_s_)
            actor.learn(b_s)
            critic.replace()
