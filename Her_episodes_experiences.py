# TODO trying different HER strategy using the distance to the object

import numpy as np
import random


class Her_rec_experiences(object):

    def __init__(self, buffer_size=1000):
        self.memory = []
        self.mem_size = buffer_size

    def add(self, experience):
        if len(self.memory) + 1 >= self.mem_size:
            self.memory[0:(1 + len(self.memory)) - self.mem_size] = []
        self.memory.append(experience)

    def sample(self, batch_size, trace_length):
        tmp_buffer = [episode for episode in self.memory if len(episode) + 1 > trace_length]
        sampled_episodes = random.sample(tmp_buffer, batch_size)
        sampled_traces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampled_traces.append(episode[point:point + trace_length])
        sampled_traces = np.array(sampled_traces)
        return np.reshape(sampled_traces, [batch_size * trace_length, 6])

    def get(self):
        return np.reshape(np.array(self.memory), [len(self.memory), 5])

    def her(self, strategy, her_samples):
        if strategy == "future":
            episode = []
            for mem_idx in range(len(self.memory)):
                episode = self.memory[mem_idx]
                for trans_idx in range(len(episode)):
                    for k in range(her_samples):
                        future_samples = np.random.randint(trans_idx, len(episode))
                        goal = episode[future_samples][3]
                        #goal_pos = goal[-3:]
                        state = episode[trans_idx][0]
                        action = episode[trans_idx][1]
                        next_state = episode[trans_idx][3]
                        #next_state_pos = next_state[-3:]
                        done = np.array_equal(goal, next_state)
                        reward = 0 if done else -1
                        episode[trans_idx][0] = state
                        episode[trans_idx][1] = action
                        episode[trans_idx][2] = reward
                        episode[trans_idx][3] = next_state
                        episode[trans_idx][4] = done
                        episode[trans_idx][5] = goal
