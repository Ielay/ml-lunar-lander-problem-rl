import os
# force TF to use CPU instead of GPU (sadly my discrete card is not support the latest CUDA version)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.models import load_model

import gym
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model = load_model('model.h5')

    env = gym.make('LunarLander-v2')

    total_rewards = []

    NUMBER_OF_EPISODES = 100
    MAX_NUMBER_OF_STEPS_IN_EPISODE = 750

    for episode_idx in range(NUMBER_OF_EPISODES):
        obs = env.reset()  # get initial state

        episode_reward = 0
        for step_idx in range(MAX_NUMBER_OF_STEPS_IN_EPISODE):
            # env.render()

            obs = np.reshape(obs, (1, 8))
            rewards = model.predict(obs)
            action = int(np.argmax(rewards[0]))

            obs, reward, done, info = env.step(action)  # step returns 4 parameters

            episode_reward += reward

            if done:
                print("Episode '%d' reward '%f'" % (episode_idx, episode_reward))
                break

        total_rewards.append(episode_reward)

    print("Average reward on test 100 games: ", np.mean(total_rewards))

    # plot test progress
    x = list(range(len(total_rewards)))
    plt.plot(x, total_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.show()

    env.close()