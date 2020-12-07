from collections import deque
import time
import random
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import nn
from envrionment import Env
from class1 import PO as Ballon
from Target import Target
import pygame

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 100_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 1  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
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





env = Env()

# For stats
ep_rewards = [-500]

# For more repetitive results


if not os.path.isdir('models'):
    os.makedirs('models')

#Agent class
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

        model.add(nn.Layer_Dense(6, 64))
        model.add(nn.Activation_ReLU())

        model.add(nn.Layer_Dense(64,64))
        model.add(nn.Activation_ReLU())
        model.add(nn.Layer_Dense(64, 64))
        model.add(nn.Activation_ReLU())







        model.add(nn.Layer_Dense(64, env.ACTION_SPACE_SIZE))
        model.add(nn.Activation_Linear())

        model.set(loss=nn.Loss_MeanSquaredError(),
                  optimizer=nn.Optimizer_Adam(learning_rate=0.001),
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

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
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


agent = DQNAgent()
averages =[]
episodes = []
minimums = []
maximums = []
# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
        pygame.init()
        clock = pygame.time.Clock()
        running = True
        screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
        pygame.display.set_caption("TheProgram")

        while running:

            episode_reward = 0
            step = 1

            # Reset environment and get initial state
            current_state = env.reset()

            # Reset flag and start iterating until episode ends
            done = False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                screen.fill((200, 200, 200))

                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, env.ACTION_SPACE_SIZE)

                new_state, reward, done = env.step(action)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                # Every step we update replay memory and train main network
                agent.update_replay_memory((current_state, action, reward, new_state, done))
                agent.train(done, step, False)

                current_state = new_state
                step += 1
                env.render(screen)
                clock.tick(60)
                pygame.display.update()
            if done:
                running = False


        pygame.quit()
    else:
        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        while not done:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, env.ACTION_SPACE_SIZE)

            new_state, reward, done = env.step(action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            showstats = False
            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                showstats = False
            else:
                pass

            agent.train(done, step, showstats)

            current_state = new_state
            step += 1
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            episodes.append(episode)
            averages.append(average_reward)
            minimums.append(min_reward)
            maximums.append(max_reward)
            print(f'episode{episode}')
            print(f'average reward: {average_reward}')
            print(f'reward min: {min_reward}')
            print(f'max reward: {max_reward}')
            print(f'epsilon: {epsilon}')
            print(f'learning rate: {agent.model.optimizer.current_learning_rate}')
            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(
                    f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')




plt.plot(episodes, minimums)
plt.plot(episodes, averages)
plt.plot(episodes, maximums)
plt.show()


agent.model.save("model8")
