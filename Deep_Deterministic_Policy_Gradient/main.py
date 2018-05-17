import gym
import numpy as np
import tensorflow as tf
import random

from ddpg import Actor, Critic
from memory import *

if __name__ == '__main__':

    env = gym.make('Pendulum-v0')
    env = env.unwrapped

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    var = 3.

    with tf.Session() as sess:
        memory = Memory(32, 10000)
        actor = Actor(sess, state_dim, action_bound, lr=0.01, tau=0.01)
        critic = Critic(sess, state_dim, actor.s, actor.s_, actor.a, actor.a_, gamma=0.9, lr=0.001, tau=0.01)
        t = critic.get_gradients()

        actor.generate_gradients(t)

        sess.run(tf.global_variables_initializer())

        for i in range(1000):
            s = env.reset()
            r_episode = 0
            for j in range(200):
                a = actor.choose_action(s)
                a = np.clip(np.random.normal(a, var), -action_bound, action_bound)  # 异策略探索
                s_, r, done, info = env.step(a)

                memory.store_transition(s, a, [r / 10], s_)

                if memory.isFull:
                    var *= 0.9995
                    b_s, b_a, b_r, b_s_ = memory.get_mini_batches()
                    critic.learn(b_s, b_a, b_r, b_s_)
                    actor.learn(b_s)

                r_episode += r
                s = s_

                if(j == 200 - 1):
                    print('episode {}\treward {:.2f}\tvar {:.2f}'.format(i, r_episode, var))
                    break
