import random
import concurrent.futures
import numpy as np
import gym
from ppo import PPO

GAMMA = 0.9
BATCH_SIZE = 200
ITER_MAX = 1000
ACTOR_NUM = 1
T_TIMESTEPS = 200


env = gym.make('Pendulum-v0')
env = env.unwrapped

state_dim = env.observation_space.shape[0]
action_bound = env.action_space.high[0]

ppo = PPO(state_dim, action_bound, 1, 0.01, 0.2, 0.001, 10)

executor = concurrent.futures.ThreadPoolExecutor(ACTOR_NUM)


def simulate():
    s = env.reset()
    r_sum = 0
    trans = []
    for step in range(T_TIMESTEPS):
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)
        trans.append([s, a, (r + 8) / 8])
        s = s_
        r_sum += r

    v_s_ = ppo.get_v(s_)
    for tran in trans[::-1]:
        v_s_ = tran[2] + GAMMA * v_s_
        tran[2] = v_s_

    return r_sum, trans


for i_iteration in range(ITER_MAX):
    futures = [executor.submit(simulate) for _ in range(ACTOR_NUM)]
    concurrent.futures.wait(futures)

    trans_with_discounted_r = []
    r_sums = []
    for f in futures:
        r_sum, trans = f.result()
        r_sums.append(r_sum)
        trans_with_discounted_r += trans

    print(i_iteration, r_sums)

    for i in range(0, len(trans_with_discounted_r), BATCH_SIZE):
        batch = trans_with_discounted_r[i:i + BATCH_SIZE]
        s, a, discounted_r = [np.array(e) for e in zip(*trans_with_discounted_r)]
        ppo.train(s, a, discounted_r[:, np.newaxis])
