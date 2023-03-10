import threading
import time
import numpy as np
import ML.agent as a
import game.board as b
import copy
import utilities.progress as progress
import matplotlib.pyplot as plt
from datetime import datetime as dt
import torch

episodes = 30000
gamma = 0.99
min_learning_rate = 0.00001
max_learning_rate = 0.1
buffer_size = 10000
batch_size = 64
epsilon_start = 1.
epsilon_end = 0.

update_net_freq = 1
update_opponent_episodes = 500
update_target_model_episodes = 200
num_agents = 4
sync_agent_episodes = 200
acc_games = 200
filename = "model_multi.pt"

accs_async = np.zeros(num_agents)
learning_rates = np.arange(min_learning_rate, max_learning_rate, step=(max_learning_rate-min_learning_rate)/num_agents)

def all_threads_finished(threads):
    for thread in threads:
        if thread.is_alive():
            return False
    
    return True

def print_threadsafe(msg:str):
    lock = threading.Lock()
    lock.acquire()
    print(msg + "\n", end="")
    lock.release()

def train_async(episode:int, i_agent:int, agent1:a.Agent, agent2:a.Agent):
    thread = threading.Thread(target=train, args=(episode, i_agent, agent1, agent2))
    thread.start()
    return thread

def train(episode_overall:int, i_agent:int, agent1:a.Agent, agent2:a.Agent):
    if i_agent == 0:
        print("train:")
        bar = progress.Bar()
    steps = 0
    for episode in range(episode_overall, episode_overall + sync_agent_episodes):
        epsilon = epsilon_start - (episode / episodes) * (epsilon_start - epsilon_end)
        board:b.ConnectFourBoard = agent1.board
        player = np.random.choice([-1, 1])

        agent1.board.reset()
        done = False
        while not done:
            state = copy.deepcopy(board.board)

            if player == 1:
                action = agent1.act(state, epsilon=epsilon, train=True)
                done, reward = board.step(action, 1)
                next_state = copy.deepcopy(board.board)
                agent1.store(state, action, reward, next_state, done)

                if done:
                    if reward == board.reward_won:
                        agent2.update_last_reward(board.reward_loose)
                    else:
                        agent2.update_last_reward(board.reward_won)
            else:
                action = agent2.act(state, epsilon=epsilon, train=True)
                done, reward = board.step(action, -1)
                next_state = copy.deepcopy(board.board)
                agent2.store(state, action, reward, next_state, done)

                if done:
                    if reward == board.reward_won:
                        agent1.update_last_reward(board.reward_loose)
                    else:
                        agent1.update_last_reward(board.reward_won)

            if steps % update_net_freq == 0:
                agent1.update_net(batch_size)

            player = -1 if player == 1 else 1
            steps += 1
            
        if episode % update_opponent_episodes == 0:
            agent2 = copy.deepcopy(agent1)
            agents[i_agent]= (agent1, agent2)

        if episode % update_target_model_episodes == 0:
            agent1.update_target_model()

        if i_agent == 0:
            bar.progress((episode-episode_overall+1) / sync_agent_episodes)
   
    agents[i_agent]= (agent1, agent2)
        #print_threadsafe("agent %s episode %s" % (i_agent, episode))

def get_acc(i_agent:int, agent1:a.Agent, agent2:a.Agent, num_games:int):
    if i_agent == 0:
        print("get_acc")
        bar = progress.Bar()
    board:b.ConnectFourBoard = agent1.board
    wins1 = 0
    wins2 = 0
    
    for i in range(num_games):
        player = np.random.choice([-1, 1])
        agent1.board.reset()
        done = False
        while not done:
            state = copy.deepcopy(board.board)

            if player == 1:
                action = agent1.act(state, epsilon=0., train=False)
                done, reward = board.step(action, 1)

                if done:
                    if reward == board.reward_won:
                        wins1 += 1
                    else:
                        wins2 += 1
            else:
                action = agent2.act(state, epsilon=1.0, train=False)
                done, reward = board.step(action, -1)

                if done:
                    if reward == board.reward_won:
                        wins2 += 1
                    else:
                        wins1 += 1

            player = -1 if player == 1 else 1
        
        if i_agent == 0:
            bar.progress((i + 1) / num_games)

    accs_async[i_agent] = wins1 / num_games

def calc_accs_async(num_games:int):
    threads = []

    for i_agent, (agent1, agent2) in enumerate(agents):
        thread = threading.Thread(target=get_acc, args=(i_agent, agent1, agent2, num_games))
        thread.start()
        threads.append(thread)

    while not all_threads_finished(threads):
        time.sleep(0.1)

def create_agents(num_agents:int):
    agents = []
    
    for i_agent in range(num_agents):
        board = b.ConnectFourBoard(render=False)
        agent1 = a.Agent(board, gamma, learning_rates[i_agent], buffer_size)
        agent1.load_model(filename)
        agent2 = a.Agent(board, gamma, learning_rates[i_agent], buffer_size)
        agents.append((agent1, agent2))
    return agents

agents = create_agents(num_agents)
episodes_plt = []
accs = []
lrs = []

if __name__ == "__main__":
    start = dt.now()
    episode = 0
    while episode < episodes:
        threads = []

        for i_agent, (agent1, agent2) in enumerate(agents):
            thread = train_async(episode, i_agent, agent1, agent2)
            threads.append(thread)

        while not all_threads_finished(threads):
            time.sleep(0.1)

        calc_accs_async(acc_games)
        i_best = np.argmax(accs_async)
        #accs_temp = [get_acc(i, agent1, agent2, acc_games) for i, (agent1, agent2) in enumerate(agents)]
        #i_best = np.argmax(accs_temp)
        for i, (agent1, agent2) in enumerate(agents):
            if i == i_best:
                continue
            agents[i][0].model = copy.deepcopy(agents[i_best][0].model)
        agents[i_best][0].save_model(filename)
        lrs.append(learning_rates[i_best])

        episode += sync_agent_episodes
        episodes_plt.append(episode)
        accs.append(accs_async[i_best])

        plt.switch_backend('agg')
        plt.figure(figsize=(12,12))
        
        plt.subplot(2, 1, 1)
        plt.title("Accuracy")
        plt.plot(episodes_plt, accs, label="accuracy")
        plt.xlabel("episode")
        plt.ylabel("accuracy")
        plt.grid()
        plt.grid(which='major', linestyle='-')
        plt.grid(which='minor', linestyle=':')
        
        plt.subplot(2, 1, 2)
        plt.title("Learning Rate")
        plt.plot(episodes_plt, lrs, label="accuracy")
        plt.xlabel("episode")
        plt.ylabel("$\eta$")
        plt.grid(which='major', linestyle='-')
        plt.grid(which='minor', linestyle=':')
        #plt.legend()
        plt.savefig("plot_multiagent.png")
        plt.close()

        print("episode {0} acc {1:.3f}\ttime: {2}\tlr: {3:.5f}".format(episode, accs_async[i_best], dt.now() - start, learning_rates[i_best]))
        
        