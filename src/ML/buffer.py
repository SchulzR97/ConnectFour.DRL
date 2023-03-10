from collections import deque
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, size:int):
        self.size = size
        self.states = deque(maxlen = size)
        self.actions = deque(maxlen = size)
        self.rewards = deque(maxlen = size)
        self.next_states = deque(maxlen = size)
        self.next_actions = deque(maxlen = size)
        self.dones = deque(maxlen = size)

    def append(self, state, action, reward, next_state, done, store_zero_reward_prop):
        if reward == 0 and np.random.random() < store_zero_reward_prop:
            return

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def sample(self, batch_size:int):
        if len(self.states) < batch_size:
            return ([],[],[],[],[])
        indices = sorted(np.random.choice(range(len(self.states)), batch_size, replace=False))
        return (torch.stack([self.states[i] for i in indices]), 
                np.array([self.actions[i] for i in indices]), 
                np.array([self.rewards[i] for i in indices]), 
                torch.stack([self.next_states[i] for i in indices]),
                np.array([self.dones[i] for i in indices]))