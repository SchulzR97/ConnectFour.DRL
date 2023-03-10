from game import board as b
import ML.agent as agent
import numpy as np
import copy
import math
import os.path
import matplotlib.pyplot as plt
import datetime as dt

def get_acc(agent1:agent.Agent, agent2:agent.Agent, board:b.ConnectFourBoard, episodes:int = 10):
    wins = 0
    
    for i in range(episodes):
        player = np.random.choice([-1, 1])
        board.reset()
        done = False
        while not done:
            if player == 1:
                action = agent1.act(board.board, 0.0, train=False)
                done, reward = board.step(action, player)
                if done and reward == board.reward_won:
                    wins += 1
            else:
                action = agent2.act(board.board, 1.0, train=False)
                done, reward = board.step(action, player)

                if done and reward == board.reward_loose:
                    wins += 1
            player = -1 if player == 1 else 1
    return wins / episodes

def moving_avg(data, period:int):
    if len(data) < period:
        return math.nan
    
    mavg = []
    for i in range(period):
        mavg.append(math.nan)
    for i in range(len(data)-period):
        mavg.append(np.average(data[i:i+period]))

    return mavg

episodes = 20000#50000
gamma = 0.9
learning_rate = 0.0001#0.0001
buffer_size = 10000#1000#10000
batch_size = 128
epsilon_start = 0.5
epsilon_end = 0.1
update_target_model_episodes = 500#100#50 # 200 funktioniert schlecht
update_oponent_episodes = 200
update_net_freq = 10
plot_freq = 20
mavg_periods = 1000
acc_tries = 1
save_model_episodes = 20
store_zero_reward_prop_start = 1.0
store_zero_reward_prop_end = 1.0

board = b.ConnectFourBoard(render=False)
agent1 = agent.Agent(board, gamma, learning_rate, buffer_size)
agent2 = agent.Agent(board, gamma, learning_rate, buffer_size)

filename = "model.pt"
plotfile = "acc.png"

local_dir = "local"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)

filename = local_dir + "/" + filename
plotfile = local_dir + "/" + plotfile
agent1.load_model(filename)

i = 0
accs_test = []
accs_train = []
wins = 0

plt.switch_backend('agg')
start = dt.datetime.now()
time_per_episode = dt.datetime.now() - start
for episode in range(episodes):
    done = False
    player = np.random.choice([-1, 1])
    board.reset()
    while not done:
        state = copy.deepcopy(board.board)
        epsilon = epsilon_start - (episode/episodes) * (epsilon_start - epsilon_end)
        store_zero_reward_prop = store_zero_reward_prop_start + (episode/episodes) * (store_zero_reward_prop_end - store_zero_reward_prop_start)
        if player == 1:
            action = agent1.act(board.board, epsilon, train=True)
            done, reward = board.step(action, player)
            next_state = copy.deepcopy(board.board)
            agent1.store(state, action, reward, next_state, done, store_zero_reward_prop)
        else:
            action = agent2.act(board.board, epsilon + 0.0, train=True)
            done, reward = board.step(action, player)
            next_state = copy.deepcopy(board.board)
            agent2.store(state, action, reward, next_state, done, store_zero_reward_prop)

        # update last reward if loose
        if reward == board.reward_won:
            if player == 1:
                agent2.update_last_reward(board.reward_loose)
            else:
                agent1.update_last_reward(board.reward_loose)
        # update last reward if won
        if reward == board.reward_loose:
            if player == 1:
                agent2.update_last_reward(board.reward_won)
            else:
                agent1.update_last_reward(board.reward_won)
        
        if done:
            if player == 1 and reward == board.reward_won or \
                player == 2 and reward == board.reward_loose:
                wins += 1

        if i%update_net_freq == 0:
            agent1.update_net(batch_size)
            #agent2.update_net(batch_size)

        player = -1 if player == 1 else 1
        i += 1
    
    acc_train = wins / (episode + 1)
    accs_train.append(acc_train)

    acc_test = get_acc(agent1, agent2, board, acc_tries)
    accs_test.append(acc_test)

    if (episode-1)%save_model_episodes == 0:
        #temp_agent = copy.deepcopy(agent1)
        #temp_agent.load_model(filename)
        agent1.save_model(filename)
        #agent1 = copy.deepcopy(temp_agent)

    if episode%update_oponent_episodes == 0:
        agent2 = copy.deepcopy(agent1)

    if episode%update_target_model_episodes == 0:
        agent1.update_target_model()

    if episode % plot_freq == 0:
        plt.figure(figsize=(12,8))

        if mavg_periods > 1:
            if mavg_periods > episode:
                mavg_test = moving_avg(accs_test, int(0.5*episode))
            else:
                mavg_test = moving_avg(accs_test, mavg_periods)
        else:
            mavg_test = accs_test
        plt.plot(mavg_test, label="test")
        plt.plot(accs_train, label="train")
        plt.title("Accuracy")
        plt.xlabel("Episode")
        plt.ylabel("accuracy")
        plt.grid()
        plt.legend()
        plt.savefig(plotfile)
        plt.close()
    
    time_per_episode = (dt.datetime.now() - start) / (episode + 1)
    remaining_time = (episodes - episode - 1) * time_per_episode
    time = dt.datetime.now() - start
    print("Episode %s acc_test: %.2f\ttime: %s, remaining time: %s" % (episode, acc_test, time, remaining_time))
