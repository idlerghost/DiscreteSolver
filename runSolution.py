# Import my funcs
from dqnAgent import DoubleDQNAgent
from playGame import playGame
from auxFuncs import env_max_score
from auxFuncs import env_sol_score

# Import the env
import gym
import tensorflow as tf

# Import whats needed by the funcs
import numpy as np
import os
import sys
import random
import math
import argparse

if __name__ == "__main__":
    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K

    K.set_session(sess)

    # Creates an argparse to get the env name needs to add parser for train or not
    parser = argparse.ArgumentParser(description='Gives args for the code')
    parser.add_argument('-e', '--env', help='Tells what env to run', required=True)
    parser.add_argument('-lr', '--learning_rate', help='Tells the lr for the env', default=0.001)
    parser.add_argument('-t', '--train', help='Tells if the agent should train. If True, trains. \
                        If False, runs', default=True)
    args = vars(parser.parse_args())

    # Run in loop:
    max_episodes = 2000
    hidden_layer = 50  # Original was 24 and 50 has a good score
    hidden_layer2 = 50
    step = 0

    env = gym.make(args['env'])
    # Get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # gets the env max score
    max_score = env_max_score(args['env'])
    sol_score = env_sol_score(args['env'])


    agent = DoubleDQNAgent(state_size, action_size, args['learning_rate'], hidden_layer,
                            hidden_layer2, args['train'])

    playGame(env, agent, max_score, args['env'], args['learning_rate'],
            max_episodes, state_size, sol_score, step, action_size)
