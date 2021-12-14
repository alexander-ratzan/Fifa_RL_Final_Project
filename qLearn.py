# Q-learning Training file that implements the following:
# Experience Replay class - creates instance that learns from previous observations     
# Authors: Michael Eve, Alex Ratzan
# Date Created: Dec 2, 2021 

#Currently the WORKING Qlearn 

# setup Qlearning training with experience replay 

import numpy as np 
import matplotlib
from matplotlib import pyplot as plt
import random
import gfootball.env as env
import absl 
import sys

def train(env, agent, num_episodes): 

    num_actions = 15  # should be 15 based on size of action_set
    max_memory = 1000  # Maximum number of experiences we are storing
    batch_size = 5  # Number of experiences we use for training per batch (OLD: 4)
    exp_replay = ExperienceReplay(max_memory=max_memory)

    win_count = 0 
    win_history = [] # list of succesful episodes

    action_set = [0,1,2,3,4,5,6,7,8,12,13,14,15,17,18]

    loss = 0.0
    
    #sys.stdout = open("outputLoss.txt", "t+a")

    win_log = [] # keeps track of scored and missed goals
    losses = [] # to visualize losses
    goals = [] # to visualize goals
    eps = list(range(num_episodes))

    ## include Win_rate at some point

    for epi in range(num_episodes): 
        print('EPISODE: ', epi)
        
        epsilon = 0.4 # 4 / ((epi + 1) ** (1 / 2)) # assuming this is epsilon decay 

        env.reset()
        done = False
        
        #call function for greedy training (exploit scoring experiences) 
        if epi % 20 == 0 and epi != 0:
            act_greedy(agent)

        ## Call TAMER here every 100 episodes 
            # call a function to visualize last reward
            # we give reward of 0.1 (TBD) to each action within the episode to then create new cumulative reward
            # options for reward key mappings: good, bad, neutral (across entire episode)

        while not done: # does entire episode with multiple actions
            observation_old, reward, done, info = env.step(0) #gets initial game state from env
            if reward == 1: 
                reward = 1000
            
            step_counter = 0

            if done:
                if reward == 0:
                    win_log.append(0)
                    break #when done == TRUE
                break

            observation_old = observation_old.reshape((1,115))

            if np.random.rand() <= epsilon:
                action = random.choice(action_set) #only working with our reduced action set 
            else: 
                # print("OBSERVATION_OLD_SHAPE: ", observation_old.shape) # expected: (115,)
                # print("OBSERVATION_OLD_TYPE: ", type(observation_old)) # reshape into 2D
                q = agent.predict(observation_old) # return output/last layer of NN - q-predict
                action = np.argmax(q[0]) # take max value, which is action to take 

            print("*****ACTION*****: ", action)
            observation_new, reward, done, info = env.step(action) #terminates when done == TRUE

            # ## Updated reward params for DNN ##
            # if reward == 0:
            #     reward = -0.1
            # if reward == 1:
            #     reward = 1000

            step_counter += 1
            observation_new = observation_new.reshape((1,115))
            if reward == 1: 
                win_count += 1
                win_history.append(epi)
                win_log.append(1)


            # store experience
            exp_replay.remember([observation_old, action, reward, observation_new], done)

            # train agent on experience: individual batches loss
            if (step_counter % 4 == 0) or done:
                # Load batch of experiences
                inputs, targets = exp_replay.get_batch(agent, batch_size=batch_size)
                # train on batches     
                batch_loss = agent.train_on_batch(inputs, targets)
                loss += batch_loss
                

        print("*******LOSS-PER-EP*****: ", loss)
        print("*******GOALS-SCORED******: ", win_count)
        losses.append(loss)
        goals.append(win_count)
    #sys.stdout.close() #save the loss in a txt file
    print("Q-LEARNING OVER!")
    print("**************************")
    print("**************************")
    
    """ Plot generation """
    ## Plot win-count over all episodes ## 
    winC_plot = plt.figure(1)
    plt.title('Win Count over all Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Win Count')
    plt.plot(eps, goals, linewidth=2.0)

    ## Plot losses over all eps ## 
    losses_plot = plt.figure(2)
    losses = np.asarray(losses)
    plt.title('Cumulative Loss over all Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Total Loss')
    plt.plot(np.arange(losses.shape[0]), losses, linewidth=2.0, color="red")

    ## Plot win_rate ## 
    plot_win_rate(win_log)

    plt.show()

    print("*******TOTAL-LOSS*****: ", loss)
    print("*******TOTAL-WIN-COUNT******: ", win_count)

def act_greedy(agent):
    greedy_env = env.create_environment(env_name='test_scenario_1v1', render=True, representation='simple115')

    greedy_env.reset()
    done = False

    while not done: 
        
        observation_old, reward, done, info = greedy_env.step(0) #gets game state from env
        if done: 
            break 
        observation_old = observation_old.reshape((1,115))
        q = agent.predict(observation_old) # return output/last layer of NN - q-predict
        action = np.argmax(q[0])
        
        observation_new, reward, done, info = greedy_env.step(action)
        observation_new = observation_new.reshape((1,115))

def plot_win_rate(win_log): 
    '''
    - take array win_log[] from train
    - compute: success_rate_l100 = [x for x in range(num_episodes)] # iterate over number of episodes and store 'average' 
                                                                    # snapshots of 100 episodes
    - loss: get from train()
    - average_reward: matters for multiple epochs (averages the win_counts from multiple)
    '''
    n = 10
    win_log = np.asarray(win_log)
    sliding_win_rate = moving_average(win_log, n)
    episode_arr = np.arange(win_log.shape[0]) # array of episodes 

    plt.figure(figsize=(12, 5))
    plt.title('Average Success Rate % over all Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Rolling Average Success Rate (sliding window = 100)')
    plt.plot(episode_arr[n - 1:], sliding_win_rate, label = 'training model') # win rate plot 
    plt.show()

def moving_average(a,n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class ExperienceReplay(object):
    
    def __init__(self, max_memory=1000, discount=0.99):
        """
        Setup
        max_memory: the maximum number of experiences we want to store
        memory: a list of experiences
        discount: the discount factor for future experience
        In the memory the information whether the game ended at the state is stored seperately in a nested array
        [...
        [experience, done]
        [experience, done]
        ...]
        """
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount


    def remember(self, states, done):
        # Save a state to memory
        self.memory.append([states, done])
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, agent, batch_size):
        len_memory = len(self.memory)

        num_actions = agent.output_shape[-1] # should be equal to size of action set
        
        env_dim = 115 # might want to check on this
        #env_dim = self.memory[0][0][0].shape[1] # might want to check on this
        inputs = np.zeros((min(len_memory, batch_size), env_dim))

        targets = np.zeros((inputs.shape[0], num_actions))

        for i, idx in enumerate(np.random.randint(0, len_memory, size = inputs.shape[0])):
            """
            Here we load one transition <s, a, r, s’> from memory
            state_t: initial state s
            action_t: action taken a
            reward_t: reward earned r
            state_tp1: the state that followed s’
            """
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            done = self.memory[idx][1]

            # inputs[i] = state_t
            inputs[i:i + 1] = state_t
            targets[i] = agent.predict(state_t)
            """
            If the game ended, the expected reward Q(s,a) should be the final reward r.
            Otherwise the target value is r + gamma * max Q(s’,a’)
            """
            Q_sa = np.max(agent.predict(state_tp1))

            # if the game ended, the reward is the final reward
            if done:  # if done is True
                targets[i, action_t] = reward_t
            else:
                # r + gamma * max Q(s’,a’)
                targets[i, action_t] = reward_t + self.discount * Q_sa

        return inputs, targets
