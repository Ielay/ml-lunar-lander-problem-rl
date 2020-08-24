import os
# force TF to use CPU instead of GPU (sadly my discrete card is not support the latest CUDA version)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import gym

from gym import envs
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

import time


# build DQN model from keras layers
def dqn_model(input_layer_size, output_layer_size, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, input_dim=input_layer_size, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_layer_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

    return model


if __name__ == "__main__":
    print(envs.registry.all())

    env = gym.make('LunarLander-v2')

    print(env.action_space)
    print(env.observation_space)

    OBSERVATION_SPACE_SIZE = env.observation_space.shape[0]
    ACTION_SPACE_SIZE = env.action_space.n

    model = dqn_model(OBSERVATION_SPACE_SIZE, ACTION_SPACE_SIZE)

    NUMBER_OF_EPISODES = 1000
    MAX_NUMBER_OF_STEPS_IN_EPISODE = 750

    BATCH_SIZE = 64

    DISCOUNT_RATE = 0.99

    exploration_rate = 1.0
    EXPLORATION_RATE_MINIMUM = 0.1
    EXPLORATION_RATE_CHANGE = (exploration_rate - EXPLORATION_RATE_MINIMUM) / ((NUMBER_OF_EPISODES / 2) * 100)

    obs_arr = np.array([])
    next_obs_arr = np.array([])
    actions_arr = np.array([], dtype=int)
    rewards_arr = np.array([])
    done_arr = np.array([])

    total_rewards = np.array([])

    for episode_idx in range(NUMBER_OF_EPISODES):
        start_time = time.time()
        # Restart the environment to start a new episode
        obs = env.reset()
        obs = np.reshape(obs, (1, OBSERVATION_SPACE_SIZE))

        episode_total_reward = 0

        explore_count = 0
        exploit_count = 0

        for step_idx in range(MAX_NUMBER_OF_STEPS_IN_EPISODE):
            # render a view with the environment current state
            # env.render()

            # choose action greedily (exploration-exploitation trade-off)
            if np.random.random() < exploration_rate:
                # explore
                explore_count += 1
                action = env.action_space.sample()
            else:
                # exploit: ask DQN to make a prediction of which action is preferred here
                exploit_count += 1
                prediction = model.predict(obs)[0]
                # print("prediction: ", prediction)
                action = np.argmax(prediction)

            # do the action
            next_obs, reward, is_done, info = env.step(action)
            next_obs = np.reshape(next_obs, (1, OBSERVATION_SPACE_SIZE))

            episode_total_reward += reward

            if len(obs_arr) == 0:
                obs_arr = np.array(obs)
                next_obs_arr = np.array(next_obs)
            else:
                obs_arr = np.vstack((obs_arr, obs))
                next_obs_arr = np.vstack((next_obs_arr, next_obs))
            actions_arr = np.append(actions_arr, action)
            rewards_arr = np.append(rewards_arr, reward)
            done_arr = np.append(done_arr, is_done)

            obs = next_obs

            if len(obs_arr) >= BATCH_SIZE:
                # print(episode_idx, ":", step_idx)
                indexes = np.random.randint(len(done_arr), size=BATCH_SIZE)

                batch_obs = np.squeeze(obs_arr[indexes])
                batch_next_obs = np.squeeze(next_obs_arr[indexes])
                batch_action = actions_arr[indexes]
                batch_reward = rewards_arr[indexes]
                batch_done = done_arr[indexes]

                all_targets = model.predict_on_batch(batch_obs)
                targets = batch_reward + DISCOUNT_RATE * (np.amax(model.predict_on_batch(batch_next_obs), axis=1)) * (1 - batch_done)

                all_targets[[np.array([i for i in range(BATCH_SIZE)])], batch_action] = targets

                # train the model
                model.fit(batch_obs, all_targets, verbose=0)

            # change exploration rate at every step until it becomes less or eq than exploration_rate_min
            # if exploration_rate > EXPLORATION_RATE_MINIMUM:
            #     exploration_rate -= EXPLORATION_RATE_CHANGE

            if exploration_rate > 0.01:
                exploration_rate *= 0.99

            # print("ep#%d:step#%d, obs=%s, reward=%s, done=%s" % (episode_idx, step_idx, observation, reward, is_done))
            if is_done:
                break

        total_rewards = np.append(total_rewards, episode_total_reward)

        print("----------------------")
        print("Episode %d reward: %d" % (episode_idx, episode_total_reward))
        print("Last step: %d" % (step_idx))
        print("Exploration rate: %f" % (exploration_rate))
        print("Explore count: %d" % (explore_count))
        print("Exploit count: %d" % (exploit_count))
        print("Time for episode: ", (time.time() - start_time), " sec.")

        # lunar lander problem is considered solved when total reward is 200+ points;
        # so we check if the last episodes was successful, the training would be stopped
        if np.mean(total_rewards[-100:]) >= 200:
            break

    # plot training progress
    x = list(range(len(total_rewards)))
    plt.plot(x, total_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.show()

    # save the model for future testing
    model.save('model.h5')

    # close gym environment
    env.close()
