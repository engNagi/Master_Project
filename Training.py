import gym
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.misc import imresize
from aux_code.DQnetwork import DQN

MAX_EXPERIENCES = 50000
MIN_EXPERIENCES = 5000
TARGET_UPDATE_PERIOD = 10000
im_height = 80
im_width = 80
action_n = 4  # env.action_space.n


def preprocess(img):
    img_temp = img[31:195]  # Choose the important area of the image
    img_temp = img_temp.mean(axis=2)  # Convert to Grayscale#
    # Downsample image using nearest neighbour interpolation
    img_temp = imresize(img_temp, size=(im_height, im_width), interp='nearest')
    return img_temp


def update_state(state, obs):
    obs_small = preprocess(obs)
    return np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)



if __name__ == '__main__':
    # hyperparameters etc
    gamma = 0.99
    batch_sz = 4
    num_episodes = 5000
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


    # Create original and target  Networks
    model = DQN(action_n=action_n, im_height=im_height, im_width=im_width, fcl_dims=512, scope="model")
    target_model = DQN(action_n=action_n, im_height=im_height, im_width=im_width, fcl_dims=512, scope="target_model")

    with tf.Session() as sess:
        model.set_session(sess)
        target_model.set_session(sess)
        sess.run(tf.global_variables_initializer())
        model.load()

        print("Filling experience replay buffer...")
        obs = env.reset()
        obs_small = preprocess(obs)
        state = np.stack([obs_small] * 4, axis=0)

        # Fill experience replay buffer
        for i in range(MIN_EXPERIENCES):
            env.render()

            action = np.random.randint(0, action_n)
            obs, reward, done, _ = env.step(action)
            next_state = update_state(state, obs)

            experience_replay_buffer.append((state, action, reward, next_state, done))

            if done:
                obs = env.reset()
                obs_small = preprocess(obs)
                state = np.stack([obs_small] * 4, axis=0)

            else:
                state = next_state

        # Play a number of episodes and learn
        for i in range(num_episodes):
            t0 = datetime.now()

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
                # Update target network
                if total_t % TARGET_UPDATE_PERIOD == 0:
                    target_model.copy_from(model)
                    print("Copied model parameters to target network. total_t = %s, period = %s" % (
                        total_t, TARGET_UPDATE_PERIOD))

                # Take action
                action = model.sample_action(state, epsilon)
                obs, reward, done, _ = env.step(action)
                obs_small = preprocess(obs)
                next_state = np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)

                episode_reward += reward

                # Remove oldest experience if replay buffer is full
                if len(experience_replay_buffer) == MAX_EXPERIENCES:
                    experience_replay_buffer.pop(0)

                # Save the recent experience
                experience_replay_buffer.append((state, action, reward, next_state, done))

                # Train the model and keep measure of time
                t0_2 = datetime.now()
                states, actions, rewards, next_states, dones = model.sample(experience_replay_buffer, batch_sz)

                targets = target_model.calculate_targets(next_states=next_states,
                                                         dones=dones,
                                                         rewards=rewards,
                                                         gamma=gamma)

                loss = model.update(states, actions, targets)

                dt = datetime.now() - t0_2

                total_time_training += dt.total_seconds()
                num_steps_in_episode += 1

                state = next_state
                total_t += 1

                epsilon = max(epsilon - epsilon_change, epsilon_min)

            duration = datetime.now() - t0

            episode_rewards[i] = episode_reward
            time_per_step = total_time_training / num_steps_in_episode

            last_100_avg = episode_rewards[max(0, i - 100):i + 1].mean()
            last_100_avgs.append(last_100_avg)
            print("Episode:", i,"Duration:", duration, "Num steps:", num_steps_in_episode,
                  "Reward:", episode_reward, "Training time per step:", "%.3f" % time_per_step,
                  "Avg Reward (Last 100):", "%.3f" % last_100_avg,"Epsilon:", "%.3f" % epsilon)

            if i % 50 == 0:
                model.save(i)
            sys.stdout.flush()

    #Plots
    plt.plot(last_100_avgs)
    plt.xlabel('episodes')
    plt.ylabel('Average Rewards')
    plt.show()
    env.close()