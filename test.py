import numpy as np
from Experience_Memory import Episode_experience
from Environment_dataset_generation import Environment
from DQnetwork import DQN
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
import sys
from ai2thor.controller import BFSController
import pandas as pd
import random

scene = "FloorPlan220"
controller = BFSController()
controller.start()
controller.search_all_closed("FloorPlan220")
reachable_position = pd.DataFrame(controller.grid_points).values

#########################   hyper-parameter

num_epochs = 500
MAX_EPISODES = 200
her_strategy = "future"
Her_samples = 4

# experience replay parameters
MAX_EXPERIENCES = 10000
ep_experience = Episode_experience()
ep_experience_her = Episode_experience()
ex_replay_buffer = []

# DQN Bathrooms parameters
batch_sz = 4
gamma = 0.99
fcl_dims = 512
optimistion_steps = 40
TARGET_UPDATE_PERIOD = 10000
total_t = 0  # counter for updating the model

#   tracking parameters
last_100_avgs = []
losses = []
episode_rewards = np.zeros(MAX_EPISODES)

#   environment initialization
env = Environment(top_view_cam=True, full_scrn=False, scene=scene)
env.make()

# Create original and target  Networks
model = DQN(action_n=6, fcl_dims=fcl_dims, scope="model")
target_model = DQN(action_n=6, fcl_dims=fcl_dims, scope="target_model")
model.load()

# epsilon for Epsilon Greedy Algorithm
epsilon = 1.0
epsilon_min = 0.1
output = []
#   main loop
with tf.Session() as sess:
    model.set_session(sess)
    target_model.set_session(sess)
    sess.run(tf.global_variables_initializer())

    #   loop for #of epochs
    for i in range(num_epochs):
        successes = 0
        total_t += 1
        print("Epochs:", i)
        #   loop for #of_cycles
        for ep in range(MAX_EPISODES):
            epsilon_change = (epsilon - epsilon_min) / 500000
            print("Episode:", ep, "Epsilon:", epsilon)
            episode_reward = 0
            num_steps_in_episode = 0
            #   parameter order returned from reset
            #   frame is removed, agent_position, done, goal, object_position
            _, state, _, _, _ = env.reset()  # reset environment
            goal = random.choice(reachable_position)  # sample a random goal from reachable position
            done = False
            while not done:
                action = model.sample_action(state, goal, epsilon)

                #   Order of variables returned form take_action method
                #   frame, agent_position, done, reward, obj_agent_dis, visible
                _, next_state, done, reward = env.take_action(action=action)

                # append to experience replay
                ep_experience.add(state, action, reward, next_state, done, goal)
                state = next_state

                episode_reward += reward

                epsilon = max(epsilon - epsilon_change, epsilon_min)

            successes += done


            if her_strategy == "future":
                print("her")
                #   HER
                for t in range(len(ep_experience.memory)):
                    for k in range(Her_samples):
                        future_samples = np.random.randint(t, len(ep_experience.memory))
                        goal = ep_experience.memory[future_samples][3]  # next_state of future
                        state = ep_experience.memory[t][0]
                        action = ep_experience.memory[t][1]
                        next_state = ep_experience.memory[t][3]
                        #   checking success for the virtual_goals
                        #   where the agent position in the next state = virtual_goal = position in the next state
                        done = np.array_equal(next_state, goal)
                        reward = 0 if done else -1
                        ep_experience_her.add(state, action, reward, next_state, done, goal)

            if len(ex_replay_buffer) > MAX_EXPERIENCES-1:
                ex_replay_buffer[-MAX_EXPERIENCES:]
            ex_replay_buffer.extend(ep_experience.memory)
            ex_replay_buffer.extend(ep_experience_her.memory)
            ep_experience.clear()
            ep_experience_her.clear()

        #   Bathrooms the DQN
        for _ in range(optimistion_steps):
            print("optimisation:")
            states, actions, rewards, next_states, dones, goal_ = model.sample(ex_replay_buffer, batch_sz)

            targets = target_model.calculate_targets(next_states=next_states,
                                                     dones=dones,
                                                     rewards=rewards,
                                                     gamma=gamma,
                                                     goals_=goal_)
            losses = 0
            loss = model.update(states, actions, targets, goal_)
            losses += loss / optimistion_steps
        #   copy target model parameters to the original model
        print("Copied model parameters to target network. total_t = %s, period = %s" % (
            total_t, TARGET_UPDATE_PERIOD))
        if total_t % TARGET_UPDATE_PERIOD == 0:
            target_model.copy_from(model)

        last_100_avg = episode_rewards[max(0, i - 100):i + 1].mean()
        last_100_avgs.append(last_100_avg)
        print("Episode:", i, "Num steps:", num_steps_in_episode, "Reward:", episode_reward, "losses:", losses,
              "Avg Reward (Last 100):", "%.3f" % last_100_avg)


        if i % optimistion_steps == 0:
            model.save(i)
        sys.stdout.flush()

    # Plots
plt.plot(last_100_avgs)
plt.xlabel('episodes')
plt.ylabel('Average Rewards')
plt.savefig("rewards.png")
plt.show()
