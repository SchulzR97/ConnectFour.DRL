import torch
import numpy as np
import copy
import os
import ML.model as cfm
import ML.buffer as erm

class Agent:
    def __init__(self, board, gamma, learning_rate, buffer_size):
        self.board = board
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.model = cfm.Model(self.board.board.shape[1])
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()#torch.nn.CrossEntropyLoss()
        self.buffer = erm.ReplayBuffer(buffer_size)
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, file_name:str):
        torch.save(self.model.state_dict(), file_name)
    
    def load_model(self, filename:str):
        if not os.path.isfile(filename):
            return
        self.model.load_state_dict(torch.load(filename))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.target_model = copy.deepcopy(self.model)

    def decode_output(self, output:torch.Tensor):
        return output.argmax().item()
    
    def update_last_reward(self, value):
        if len(self.buffer.rewards) == 0:
            return
        self.buffer.rewards[-1] = value
    
    def act(self, state, epsilon, train:bool):
        if len(state.shape) == 2:
            state = state.reshape(1, state.shape[0], state.shape[1])

        action = 0
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.board.board.shape[1])
        else:
            if not train:
                self.model.eval()
                with torch.no_grad():
                    action = self.model(state)
                    action = self.decode_output(action)
            else:
                self.model.train()
                action = self.model(state)
                action = self.decode_output(action)

        return action

    def store(self, state, action, reward, next_state, done, store_zero_reward_prop):
        self.buffer.append(state, action, reward, next_state, done, store_zero_reward_prop)

    def update_net(self, batch_size):
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        if len(states) == 0:
            return
        
        states = states.reshape(batch_size, 1, states.shape[1], states.shape[2])
        next_states = next_states.reshape(batch_size, 1, next_states.shape[1], next_states.shape[2])

        with torch.no_grad():
            q_values = self.model(states)# torch.zeros((batch_size, self.board.board.shape[1]))
            next_q_values = self.target_model(next_states)#torch.zeros((batch_size, self.board.board.shape[1]))
        next_max_q_values = torch.zeros((batch_size, 1))
        targets = q_values#torch.zeros((batch_size, self.board.board.shape[1]))
        for i in range(batch_size):
            #state = states[i].reshape((1, self.board.board.shape[0], self.board.board.shape[1]))
            #next_state = next_states[i].reshape((1, self.board.board.shape[0], self.board.board.shape[1]))
            #with torch.no_grad():
                #q_value = self.model(state).detach()
                #q_values[i] = q_value
                #next_q_value = self.target_model(next_state).detach()
                #next_q_values[i] = next_q_value
            if dones[i] == True:
                next_max_q_values[i] = 0.#-1-
            else:
                next_max_q_values[i] = torch.max(next_q_values[i])

                #targets[i] = q_values[i]
            #targets[i, actions[i]] = rewards[i] + self.gamma * next_max_q_values[i]
            #targets[i, actions[i]] = q_values[i, actions[i]] + self.learning_rate * (rewards[i] + self.gamma * next_max_q_values[i] - q_values[i, actions[i]])
            targets[i, actions[i]] = q_values[i, actions[i]] + (rewards[i] + self.gamma * next_max_q_values[i] - q_values[i, actions[i]])

        self.model.train()

        predictions = self.model(states)
        loss = self.criterion(predictions, targets)
        
        self.optimizer.zero_grad()
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
