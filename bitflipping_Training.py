import numpy as np
from Experience_Memory import Episode_experience
from DQnetwork_bitflipping import DQN
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from bitflipping import BitFlip

#########   bitfliping environment
size = 50
env = BitFlip(reward_type="sparse", n=size)

#########################   hyper-parameter
num_epochs = 2000




num_episodes = 16
her_strategy = "future"
Her_samples = 4

# experience replay parameters
MAX_EXPERIENCES = 10000
ep_experience = Episode_experience()
ex_replay_buffer = []

# DQN Bathrooms parameters
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
epsilon_min = 0.02
epsilon_decay = 0.95
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
                next_state, reward, done = env.step(action=action)

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
                        future_samples = np.random.randint(t, len(ep_experience.memory)) # index of the future transitiobn
                        #future_samples_idx = ep_experience.memory[t+Her_samples]
                        goal = ep_experience.memory[future_samples][3]  # next_state of the future transition
                        state = ep_experience.memory[t][0]
                        action = ep_experience.memory[t][1]
                        next_state = ep_experience.memory[t][3]
                        done = np.array_equal(next_state, goal)
                        reward = 0 if done else -1
                        ep_experience.add(state, action, reward, next_state, done, goal)

            # Remove oldest experience if replay buffer is full
            model.buffer.extend(ep_experience.memory)
            ep_experience.clear()
        #   Bathrooms the DQN
        mean_loss = model.optimize(model=model, target_model=target_model,
                                   optimization_steps=optimistion_steps,
                                   batch_size=batch_sz)

        # if i % 75 == 0:
        print("update Target network")
        target_model.soft_update_from(model)

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        #   epsilon decay
        losses.append(mean_loss)
        success_rate.append(successes / num_episodes)
        print("\repoch", i + 1, "success rate", success_rate[-1], 'loss %.2f:' % losses[-1],
              'exploration', epsilon, end=' ' * 10)

        if i % 50 == 0:
            print("Saving the model")
            model.save(i)
        sys.stdout.flush()

    # Plots
plt.plot(losses)
plt.xlabel('episodes')
plt.ylabel('Losses')
plt.savefig("losses.png")
plt.show()

plt.plot(success_rate)
plt.xlabel('episodes')
plt.ylabel('Success rate')
plt.savefig("Success.png")
plt.show()
