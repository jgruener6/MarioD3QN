import os
import random
from collections import deque
import gym
import gym_super_mario_bros.actions as actions
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential, Model, clone_model
from keras.layers import Conv2D, Dense, Flatten, Input, Lambda
from keras.optimizers import Adam
from keras import backend as K
from wrappers import wrap_nes
from utils import plotLearning

class ReplyBuffer:
    def __init__(self, memory_size=20000):
        self.state = deque(maxlen=memory_size)
        self.action = deque(maxlen=memory_size)
        self.reward = deque(maxlen=memory_size)
        self.next_state = deque(maxlen=memory_size)
        self.done = deque(maxlen=memory_size)

    def append(self, state, action, reward, next_state, done):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.done.append(done)

    def __len__(self):
        return len(self.done)

class Agent:
    def __init__(self, env, memory_size=20000):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.observation_shape = env.observation_space.shape
        self.memory = ReplyBuffer(memory_size=memory_size)
        self.batch_size = 32
        self.update_frequency = 4
        self.tau = 1000
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.0001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        input_shape = self.observation_shape

        input_layer = Input(shape=input_shape)
        conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_layer)
        conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
        conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
        flattened = Flatten()(conv3)

        # Advantage streama
        fc_advantage = Dense(512, activation='elu', kernel_initializer='random_uniform')(flattened)
        advantage = Dense(self.action_size, activation='linear')(fc_advantage)

        # Value stream
        fc_value = Dense(512, activation='elu', kernel_initializer='random_uniform')(flattened)
        value = Dense(1, activation='linear')(fc_value)

        # Combine advantage and value to get final Q-values
        # def dueling_operator(advantage, value):
        #     return value + advantage - K.mean(advantage, axis=1, keepdims=True)

# Replace the Lambda layer in _build_model method
        combined = Lambda(lambda x: x[1] + x[0] - K.mean(x[0], axis=1, keepdims=True), output_shape=(self.action_size,))([advantage, value])

        model = Model(inputs=input_layer, outputs=combined)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_network(self):
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state)[0])

    def experience_reply(self):
        if self.batch_size > len(self.memory):
            return

        indices = np.random.choice(range(len(self.memory)), size=self.batch_size)
        state_sample = np.array([self.memory.state[i][0] for i in indices])
        action_sample = np.array([self.memory.action[i] for i in indices])
        reward_sample = np.array([self.memory.reward[i] for i in indices])
        next_state_sample = np.array([self.memory.next_state[i][0] for i in indices])
        done_sample = np.array([self.memory.done[i] for i in indices])

        target = self.model.predict(state_sample)
        target_next = self.target_model.predict(next_state_sample)

        for i in range(self.batch_size):
            if done_sample[i]:
                target[i][action_sample[i]] = reward_sample[i]
            else:
                target[i][action_sample[i]] = reward_sample[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(
            np.array(state_sample),
            np.array(target),
            batch_size=self.batch_size,
            verbose=0
        )

    def load_weights(self, weights_file):
        self.epsilon = self.epsilon_min
        self.model.load_weights(weights_file)

    def save_weights(self, weights_file):
        self.model.save_weights(weights_file)


if __name__ == "__main__":
    try:
        monitor = True
        env = wrap_nes("SuperMarioBros-1-2-v0", actions.SIMPLE_MOVEMENT)

        if monitor:
            env = gym.wrappers.Monitor(env, "recording", video_callable=lambda episode_id: True,  force=True)

        num_episodes = 500
        num_episode_steps = env.spec.max_episode_steps
        frame_count = 0
        max_reward = 0

        agent = Agent(env=env, memory_size=20000)

        if os.path.isfile("super_mario_bros_v0.h5"):
            agent.load_weights("super_mario_bros_v0 copy.h5")

        scores =[]
        eps_hist = []
        counter = 0
        for episode in range(num_episodes):
            total_reward = 0
            observation = env.reset()
            state = np.reshape(observation, (1,) + env.observation_space.shape)

            for episode_step in range(num_episode_steps):
                env.render(mode="human")
                action = agent.act(state)
                observation, reward, done, _ = env.step(action)
                total_reward += reward
                next_state = np.reshape(observation, (1,) + env.observation_space.shape)
                agent.memorize(state, action, reward, next_state, done)

                if frame_count % agent.update_frequency == 0:
                    agent.experience_reply()

                if frame_count % agent.tau == 0:
                    agent.update_target_network()

                state = next_state
                frame_count += 1

                if done:
                    print("Episode %d/%d finished after %d episode steps with total reward = %f."
                        % (episode + 1, num_episodes, episode_step + 1, total_reward))
                    if total_reward >=300:
                        print("\n\n\n\n\n OVER 300 \n\n\n\n\n\n\n")
                    break

                elif episode_step >= num_episode_steps - 1:
                    print("Episode %d/%d timed out at %d with total reward = %f."
                        % (episode + 1, num_episodes, episode_step + 1, total_reward))

            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            eps_hist.append(agent.epsilon)
            scores.append(reward)

            if total_reward > max_reward:
                agent.save_weights("super_mario_bros_v0.h5")
                keras.backend.clear_session()
            counter +=1

        # Closes the environment
        
        # plt.plot(scores)
        # plt.xlabel("Episodes")
        # plt.ylabel("Scores")
        # plt.show()
        env.close()

    except KeyboardInterrupt:
        print("done")
        # plt.plot(scores)
        # plt.xlabel("Episodes")
        # plt.ylabel("Scores")
        # plt.show()



### Episode 4433/50000 finished after 443 episode steps with total reward = 316.900000.s