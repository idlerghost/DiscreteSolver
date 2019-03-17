from memory import Memory
from keras.utils import to_categorical
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from collections import deque
import numpy as np
import random

class DoubleDQNAgent:
    def __init__(self, state_size, action_size, learning_rate, hidden_layer, hidden_layer2,
                 trainfactor, use_decay = False, use_amsgrad = True):
        # If the network is already trained, set to False
        self.train = trainfactor
        # If its desired to see the env animation, set to True
        self.render = False
        # Get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layer = hidden_layer
        self.hidden_layer2 = hidden_layer2

        # These are hyper parameters for the Double DQN
        self.discount_factor = 0.99  # also known as gamma
        self.learning_rate = learning_rate  # 0.001 for Adam, 0.002 if Nadam or Adamax
        if (self.train):
            self.epsilon = 1.0  # if we are training, exploration rate is set to max
        else:
            self.epsilon = 1e-6  # if we are just running, exploration rate is min
        self.epsilon_decay = 0.999
        self.epsilon_min = 1e-6
        self.batch_size = 64
        # This should be action_size * state_size * multiplier
        self.train_start = int(action_size ** action_size * 250)   # when will we start training, maybe some higher numbers. Default is 1000
        self.use_amsgrad = use_amsgrad
        self.use_decay = use_decay
        self.t = 1# creates a time variable to use in learning rate decay
        # Create replay memory using deque
        self.memory_size = int(action_size ** action_size * 750)
        self.memory = Memory(self.memory_size)

        # Create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()



        # Initialize target model so that the parameters of model and target model are the same
        self.update_target_model()

    # Aproximate Q function using Neural Network
    # State is input and Q value of each action is output of the network
    def build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_layer, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.hidden_layer2, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.hidden_layer, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        # Probably use decay = self.learning_rate/root(t) or just /root(t)
        if(self.use_decay):
            optimizer = Adam(lr=self.learning_rate, decay= self
                             .learning_rate /math.sqrt(self.t),
                             amsgrad= self.use_amsgrad)  # try Adamax, Adam and Nadam
        else:
            optimizer = Adam(lr=self.learning_rate,amsgrad= self.use_amsgrad)  # try Adamax, Adam and Nadam
        # try logcosh loss
        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy', 'mean_squared_error'])
        return model

    # Updates the target model to be the same as the model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        if self.epsilon ==  1:
            done = True

        target = self.model.predict([state])
        old_val = target[0][action]
        target_val = self.target_model.predict([next_state])
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.discount_factor*(np.amax(target_val[0]))
        error = abs(old_val - target[0][action])

        self.memory.add(error, (state, action, reward, next_state, done))

    # Save sample <s, a, r, s'> to the replay memory
    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Pick random samples from the replay memory
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = self.memory.sample(self.batch_size)

        errors = np.zeros(self.batch_size)
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [],[],[]

        for i in range(self.batch_size):
            states[i] = mini_batch[i][1][0]
            actions.append(mini_batch[i][1][1])
            rewards.append(mini_batch[i][1][2])
            next_states[i] = mini_batch[i][1][3]
            dones.append(mini_batch[i][1][4])

        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            old_val = target[i][actions[i]]
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * np.amax(target_val[i])

            errors[i]  = abs(old_val - target_val[i][actions[i]])

        for i in range(self.batch_size):
            idx = mini_batch[i][0]
            self.memory.update(idx, errors[i])

        self.model.train_on_batch(states, target)#, callbacks=[PlotLossesCallback()])

    # Load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # Save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)

    def update_step(self, step):
        self.t = step