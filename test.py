import numpy as np
from Experience_Memory import Episode_experience
from Environment import Environment
from DQnetwork import DQN
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys


#########################   hyper-parameter
num_epochs = 200
num_episodes = 100
her_strategy = "future"
Her_samples = 4

# experience replay parameters
MAX_EXPERIENCES = 10000
ep_experience = Episode_experience()
ep_experience_her = Episode_experience()
ex_replay_buffer = []

# DQN training parameters
batch_sz = 4
gamma = 0.99
fcl_dims = 512
optimistion_steps = 40
TARGET_UPDATE_PERIOD = 10000
total_t = 0  # counter for updating the model

#   tracking parameters
episode_rewards = np.zeros(num_episodes)
last_100_avgs = []
losses = []

#   environment initialization
env =Environment(object_name="Television")

# Create original and target  Networks
model = DQN(action_n=6, fcl_dims=fcl_dims, scope="model")
target_model = DQN(action_n=6, fcl_dims=fcl_dims, scope="target_model")

# epsilon for Epsilon Greedy Algorithm
epsilon = 1.0
epsilon_min = 0.1
epsilon_change = (epsilon - epsilon_min) / 500000

#   main loop
with tf.Session() as sess:
    model.set_session(sess)
    target_model.set_session(sess)
    sess.run(tf.global_variables_initializer())

    #   loop for #of epochs
    for i in tqdm(range(num_epochs), total=num_epochs):
        successes = 0
        total_t += 1

        #   loop for #of_cycles
        for n in range(num_episodes):
            episode_reward = 0
            num_steps_in_episode = 0
            #   parameter order returned from reset
            #   frame is removed, agent_position, done, goal, object_position
            _, state, done, goal, _ = env.reset()  # reset environment

            done = False
            while not done:
                state = np.concatenate((state, goal), axis=0)  # state shape = 1*6
                action = model.sample_action(state, epsilon)

                #   Order of variables returned form take_action method
                #   frame, agent_position, done, reward, obj_agent_dis, visible
                next_state, done, reward, _, _ = env.take_action(action=action)

                # append to experience replay
                ep_experience.add(state, action, reward, next_state, done)
                state = next_state

                episode_reward += reward

            successes += done

            if her_strategy == "future":
                #   HER
                for t in range(len(ep_experience.memory)):
                    for k in range(Her_samples):
                        future_samples = np.random.randint(t, len(ep_experience.memory))
                        goal = ep_experience.memory[future_samples][3]  # next_state of future
                        state = ep_experience.memory[t][0]
                        action = ep_experience.memory[t][1]
                        next_state = ep_experience.memory[t][3]
                        state = np.concatenate((state, goal), axis=0)
                        next_state = np.concatenate((next_state, goal), axis=0)
                        #   checking success for the virtual_goals
                        #   where the agent position in the next state = virtual_goal = position in the next state
                        done = np.array_equal(next_state, goal)
                        reward = 0 if done else -1
                        ep_experience_her.add(state, action, reward, next_state, done)

            # Remove oldest experience if replay buffer is full
            if len(ex_replay_buffer) == MAX_EXPERIENCES:
                del ex_replay_buffer[0:2]

            ex_replay_buffer.append(ep_experience.memory)
            ex_replay_buffer.append(ep_experience_her.memory)
            ep_experience.clear()
            ep_experience_her.clear()

        #   training the DQN
        for _ in range(optimistion_steps):
            states, actions, rewards, next_states, dones = model.sample(ex_replay_buffer, batch_sz)

            targets = target_model.calculate_targets(next_states=next_states,
                                                     dones=dones,
                                                     rewards=rewards,
                                                     gamma=gamma)
            losses = 0
            loss = model.update(states, actions, targets)
            losses += loss/optimistion_steps
        #   copy target model parameters to the original model
        if total_t % TARGET_UPDATE_PERIOD == 0:
            target_model.copy_from(model)
            print("Copied model parameters to target network. total_t = %s, period = %s" % (
                total_t, TARGET_UPDATE_PERIOD))

        #   epsilon decay
        epsilon = max(epsilon - epsilon_change, epsilon_min)

        episode_rewards[i] = episode_reward

        last_100_avg = episode_rewards[max(0, i - 100):i + 1].mean()
        last_100_avgs.append(last_100_avg)
        print("Episode:", i, "Num steps:", num_steps_in_episode, "Reward:", episode_reward, "loss", losses[-1],
              "Avg Reward (Last 100):", "%.3f" % last_100_avg, "Epsilon:", "%.3f" % epsilon)

        if i % 50 == 0:
            model.save(i)
        sys.stdout.flush()

    # Plots
plt.plot(last_100_avgs)
plt.xlabel('episodes')
plt.ylabel('Average Rewards')
plt.savefig("rewards.png")
plt.show()
