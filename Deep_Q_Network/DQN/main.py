import gym
import numpy as np
import tensorflow as tf
from dqn import *

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = env.unwrapped

    with tf.Session() as sess:
        rl = DQN(
            sess=sess,
            s_dim=env.observation_space.shape[0],
            a_dim=env.action_space.n,
            batch_size=128,
            gamma=0.99,
            lr=0.01,
            epsilon=0.1,
            replace_target_iter=300
        )
        tf.global_variables_initializer().run()

        rs = []
        for i_episode in range(1000):

            s = env.reset()
            r_sum = 0
            while True:
                a = rl.choose_action(s)

                s_, r, done, _ = env.step(a)

                rl.store_transition_and_learn(s, a, r, s_, done)

                r_sum += 1
                if done:
                    print(i_episode, r_sum)
                    rs.append(r_sum)
                    break

                s = s_

        print('mean', np.mean(rs))
