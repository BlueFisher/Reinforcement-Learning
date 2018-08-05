import random
import collections
import numpy as np


class Memory(object):
    def __init__(self, batch_size, max_size):
        self.batch_size = batch_size
        self.max_size = max_size
        self.isFull = False
        self.episodes = collections.deque()

    # s: (num_steps, s_dim)
    # a: (num_steps, a_dim)
    # r: (num_steps, 1)
    # s_: (num_steps, s_dim)
    def store_transition(self, s, a, r, s_):
        if len(self.episodes) == self.max_size:
            self.episodes.popleft()

        self.episodes.append((s, a, r, s_))
        if len(self.episodes) >= self.max_size:
            self.isFull = True

    def can_batch(self):
        return len(self.episodes) >= self.batch_size

    def get_mini_batches(self):
        t = random.sample(self.episodes, k=self.batch_size)

        # s: (batch_size, num_steps, s_dim)
        # a: (batch_size, num_steps, a_dim)
        # r: (batch_size, num_steps, 1)
        # s_: (batch_size, num_steps, s_dim)
        return tuple(np.array(e) for e in zip(*t))
