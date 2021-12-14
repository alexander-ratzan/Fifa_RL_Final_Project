# Main file for google-ai DQN-TAMER implementation 
# Authors: Michael Eve, Alex Ratzan
# Date Created: Dec 2, 2021 

import numpy as np
import gfootball.env as env
import qLearn as ql
import keras
import time 

from absl import flags 
from keras.models import tf
from keras.layers.core import Dense
from keras.models import Sequential
from keras.models import model_from_json
from tensorflow.keras.optimizers import SGD

import keras

def main(): 
    env1 = env.create_environment(env_name='test_scenario_1v1', render=False, representation='simple115')

    # initialize model to be used 
    grid_size = 115 
    num_actions = 19
    # hidden_size = 512 >> Not used anymore 
    num_episodes = 30  

    agent = baseline_model(grid_size, num_actions)

    ql.train(env1, agent, num_episodes)

    print("TRAINING DONE!")
    print(agent.summary())

def baseline_model(grid_size, num_actions):
    # setting up the model with keras
    #init = tf.keras.initializers.HeUniform()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(grid_size,)))
    model.add(tf.keras.layers.Dense(32, activation='relu')) #, kernel_initializer=init))
    model.add(tf.keras.layers.Dense(32)) # kernel_initializer=init))
    model.add(tf.keras.layers.Dense(num_actions))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(lr=0.001))

    return model

if __name__ == "__main__":
    main()

