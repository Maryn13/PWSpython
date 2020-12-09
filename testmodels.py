
import numpy as np

from envrionment import Env
import matplotlib.pyplot as plt
import pygame
from collections import deque
import time
import random
import nn
from tqdm import tqdm
print("hi")
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 100_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 1  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = 0  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 2000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99997
MIN_EPSILON = 0.01

#  Stats settings
AGGREGATE_STATS_EVERY = 10 # episodes
SHOW_PREVIEW = False
class DQNAgent:
    def __init__(self, inputrate = 0.2):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_parameters(self.model.get_parameters())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.inputrate = inputrate

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = nn.Model()

        model.add(nn.Layer_Dense(6, 128))
        model.add(nn.Activation_ReLU())

        model.add(nn.Layer_Dense(128,128))
        model.add(nn.Activation_ReLU())
        model.add(nn.Layer_Dense(128, 128))
        model.add(nn.Activation_ReLU())







        model.add(nn.Layer_Dense(128, env.ACTION_SPACE_SIZE))
        model.add(nn.Activation_Linear())

        model.set(loss=nn.Loss_MeanSquaredError(),
                  optimizer=nn.Optimizer_Adam(learning_rate=0.0005),
                  accuracy=nn.Accuracy_Regression())
        model.finalize()
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step, stats):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])

        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)
        future_q_action = self.model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                future_action = np.argmax(future_q_action[index])

                max_future_q = future_qs_list[index][future_action]
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state

            current_qs = current_qs_list[index]

            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        y = np.array(y)
        # Fit on all samples as one batch, log only on terminal state
        self.model.train(np.array(X), y, epochs=1, batch_size=MINIBATCH_SIZE, show_stats=stats)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        self.update_target(True,  0.001)

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state))

    def update_target(self, softupdate, tau):
        if not softupdate:
            if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.set_parameters(self.model.get_parameters())
                self.target_update_counter = 0
            return
        if softupdate:
            q_model_theta = self.model.get_parameters()
            target_model_theta = self.target_model.get_parameters()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = list(target_weight)
                q_weight = list(q_weight)

                target_weight[0] = target_weight[0] * (1 - tau) + q_weight[0] * tau
                target_weight[1] = target_weight[1] * (1 - tau) + q_weight[1] * tau
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_parameters(target_model_theta)


env = Env()
num_games = 2
load_checkpoint = True

agent = DQNAgent()


agent.model.set_parameters(nn.Model.load('models2/beste_model-1350-719-150.model').get_parameters())
agent.target_model.set_parameters(agent.model.get_parameters())

filename = 'LunarLander-Dueling-DDQN-512-Adam-lr0005-replace100.png'
scores = []
eps_history = []
n_steps = 0

for i in range(num_games):

    done = False
    observation = env.reset()
    score = 0
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
    pygame.display.set_caption("TheProgram")
    while not done:
        events = pygame.event.get()

        for event in events:

            if event.type == pygame.QUIT:
                done = True
        screen.fill((200, 200, 200))
        action = np.argmax(agent.get_qs(observation))
        observation_, reward, done = env.step(action)
        score += reward
        agent.update_replay_memory((observation, action,
                                    reward, observation_, done))

        print(reward)

        env.render(screen)

        observation = observation_
        clock.tick(60)
        pygame.display.update()
    pygame.quit()
    scores.append(score)
    avg_score = np.mean(scores[max(0, i-100):(i+1)])
    print('episode: ', i,'score %.1f ' % score,
             ' average score %.1f' % avg_score)




x = [i+1 for i in range(num_games)]

plt.plot(x, scores)
plt.show()
