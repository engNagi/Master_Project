import gym
import sys
import random
import numpy as np
import tensorflow as tf
from scipy.misc import imresize
from DQnetwork import DQN

MAX_EXPERIENCES = 50000
MIN_EXPERIENCES = 5000
im_height = 80
im_width = 80
action_n = 4  # env.action_space.n


def preprocess(img):
    # img_temp = img[31:195]  # Choose the important area of the image
    img_temp = img.mean(axis=2)  # Convert to Grayscale#
    # Downsample image using nearest neighbour interpolation
    # img_temp = imresize(img_temp, size=(im_height, im_width), interp='nearest')
    return img_temp


def update_state(state, obs):
    obs_small = preprocess(obs)
    return np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)


if __name__ == '__main__':
    # hyperparameters etc
    gamma = 0.99
    batch_sz = 4
    num_episodes = 2500
    total_t = 0
    experience_replay_buffer = []
    episode_rewards = np.zeros(num_episodes)
    last_100_avgs = []

    # epsilon for Epsilon Greedy Algorithm
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) / 500000

    # Create Atari Environment
    env = gym.envs.make("Breakout-v0")

    target_model = DQN(action_n=action_n, im_height=im_height, im_width=im_width, fcl_dims=512, scope="target_model")

    with tf.Session() as sess:
        target_model.set_session(sess)
        sess.run(tf.global_variables_initializer())
        target_model.load()

        for i in range(num_episodes):
            total_rewards = 0
            episode = +1
            # Reset the environment
            obs = env.reset()
            obs_small = preprocess(obs)
            state = np.stack([obs_small] * 4, axis=0)
            assert (state.shape == (4, 80, 80))
            loss = None

            total_time_training = 0
            num_steps_in_episode = 0
            episode_reward = 0

            done = False
            while not done:
                env.render()

                # Take action
                action = target_model.sample_action(state, epsilon)
                obs, reward, done, _ = env.step(action, )
                obs_small = preprocess(obs)
                next_state = np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)

                episode_reward += reward
                num_steps_in_episode += 1

                if done:
                    print("Episode:", episode, "Score", total_rewards)
                    break
                state = next_state
