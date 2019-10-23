import numpy as np
from Experience_Memory import Episode_experience
from DQnetwork_bitflipping import DQN
import tensorflow as tf
# import matplotlib.pyplot as plt
import sys
from bitflipping import BitFlip

#########   bitfliping environment
size = 30
env = BitFlip(reward_type="sparse", n=size)

#########################   hyper-parameter
num_epochs = 250
num_episodes = 16
her_strategy = "future"
Her_samples = 4

# experience replay parameters
MAX_EXPERIENCES = 10000
ep_experience = Episode_experience()
ep_experience_her = Episode_experience()
ex_replay_buffer = []

# DQN training parameters
batch_sz = 128
gamma = 0.99
fcl_dims = 256
optimistion_steps = 40
TARGET_UPDATE_PERIOD = 10000

#   tracking parameters
losses = []
mean_loss = 0
loss_ = 0
success_rate = []

#   environment initialization
# env = Environment(object_name="Television")

# Create original and target  Networks
model = DQN(action_n=size, fcl_dims=fcl_dims, scope="model")
target_model = DQN(action_n=size, fcl_dims=fcl_dims, scope="target_model")
model.load()

# epsilon for Epsilon Greedy Algorithm
epsilon = 0.2
epsilon_min = 0.001
epsilon_change = (epsilon - epsilon_min) / 500

#   main loop
with tf.Session() as sess:
    model.set_session(sess)
    target_model.set_session(sess)
    sess.run(tf.global_variables_initializer())

    #   loop for #of epochs
    for i in range(num_epochs):
        successes = 0
        #   loop for #of_cycles
        for n in range(num_episodes):
            state, goal = env.reset()  # reset environment
            for t in range(size):
                action = model.sample_action(state, goal, epsilon)
                #   Order of variables returned form take_action method
                #   frame, agent_position, done, reward, obj_agent_dis, visible
                next_state, done, reward = env.step(action=action)

                # append to experience replay
                ep_experience.add(state, action, reward, next_state, done, goal)
                state = next_state
                if done:
                    break
            successes += done
            # episode_reward += reward

            if her_strategy == "future":
                #   HER
                for t in range(len(ep_experience.memory)):
                    for k in range(Her_samples):
                        future_samples = np.random.randint(t, len(ep_experience.memory))
                        goal = ep_experience.memory[future_samples][3]  # next_state of future
                        state = ep_experience.memory[t][0]
                        action = ep_experience.memory[t][1]
                        next_state = ep_experience.memory[t][3]
                        done = np.array_equal(goal, next_state)
                        reward = 0 if done else -1
                        ep_experience.add(state, action, reward, next_state, done, goal)
            # Remove oldest experience if replay buffer is full
            model.remember(ep_experience.memory)
            ep_experience.clear()
        #   training the DQN
        mean_loss = model.optimize(model=model, target_model=target_model, buffer=ex_replay_buffer,
                             optimization_steps=optimistion_steps,
                             batch_size=batch_sz, gamma=gamma)
        target_model.copy_from(model)

        #   epsilon decay
        epsilon = max(epsilon - epsilon_change, epsilon_min)
        losses.append(mean_loss)
        success_rate.append(successes / num_episodes)
        print("\repoch", i + 1, "success rate", success_rate[-1], 'loss %.2f:' % losses[-1],
              'exploration', epsilon)

        if i % 50 == 0:
            print("Saving the model")
            model.save(i)
        sys.stdout.flush()

    # Plots
# plt.plot(losses)
# plt.xlabel('episodes')
# plt.ylabel('Losses')
# plt.savefig("losses.png")
# plt.show()
