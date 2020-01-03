import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from autoencoder import load_autoencoder
from DRQN_HER import DRQN
from Her_episodes_experiences import Her_episodes_experiences
from Environment_dataset_generation import Environment

random.seed(123)
np.random.seed(123)

#########   bitfliping environment
action_n = 6
#########################   hyper-parameter
num_episodes = 10000
max_episode_length = 50
her_strategy = "future"
her_samples = 4

# experience replay parameters
MAX_EXPERIENCES = 10000
ep_experience = Her_episodes_experiences()

# DQN Bathrooms parameters
batch_size = 4
trace_length = 8
gamma = 0.99
fcl_dims = 256
nodes_num = 518
optimistion_steps = 40
TARGET_UPDATE_PERIOD = 10000

#   tracking parameters
losses = []
mean_loss = 0
loss_ = 0
success_rate = []
episode_reward = 0

#   environment initialization
env = Environment(random_goals=True)
env.make()

#   Autoenconder
print("Autoencoder")
ae_sess, ae = load_autoencoder()

# Create original and target  Networks


# epsilon for Epsilon Greedy Algorithm
epsilon = 0.2
epsilon_min = 0.02
epsilon_decay = 0.95
epsilon_change = (epsilon - epsilon_min) / 500

#   main loop
print("DQN_HER_Model")
drqn_graph = tf.Graph()
with drqn_graph.as_default():
    model = DRQN(action_n=action_n, nodes_num=518, fcl_dims=nodes_num, scope="model")
    target_model = DRQN(action_n=action_n, nodes_num=518, fcl_dims=nodes_num, scope="target_model")

drqn_sess = tf.Session(graph=drqn_graph)
model.set_session(drqn_sess)
target_model.set_session(drqn_sess)
with drqn_sess.as_default():
    with drqn_graph.as_default():
        tf.global_variables_initializer().run()
        model.load()

        # loop for #of_cycles
        successes = 0
        for n in range(num_episodes):
            # reset environment
            obs_state, pos_state, goal, obj_agent_dis, _, _ = env.reset()

            features = ae_sess.run(ae.feature_vector, feed_dict={ae.image: obs_state[None, :, :, :]})
            features = np.squeeze(features, axis=0)
            obs_pos_state = np.concatenate((features, pos_state), axis=0)

            rnn_state = (np.zeros([1, nodes_num]), np.zeros([1, nodes_num]))
            done = False
            i = 0
            while i < max_episode_length:
                i += 1
                action, rnn_state_ = model.sample_action(goal=goal,
                                                         batch_size=1,
                                                         trace_length=1,
                                                         epsilon=epsilon,
                                                         rnn_state=rnn_state,
                                                         obs_pos_state=obs_pos_state)

                obs_state_, pos_state_, done, reward, object_agent_dis_, visible, _ = env.step(action, obj_agent_dis)

                features_, ae_summary = ae_sess.run([ae.feature_vector, ae.merged],
                                                    feed_dict={ae.image: obs_state[None, :, :, :]})
                features_ = np.squeeze(features_, axis=0)
                obs_pos_state_ = np.concatenate((features_, pos_state_), axis=0)

                # append to experience replay
                ep_experience.add(obs_pos_state, action, reward, obs_pos_state_, done, goal)

                rnn_state = rnn_state_
                obs_pos_state = obs_pos_state_
                obj_agent_dis = object_agent_dis_

                if done:
                    break
            if visible:
                successes += done
            episode_reward += reward

            ep_memory = ep_experience.her(strategy="future", her_samples=her_samples)

            model.buffer.extend(ep_memory)
            ep_experience.clear()

            if n % 2:
                print("optimizing")
                mean_loss, drqn_summary = model.optimize(model=model,
                                                         batch_size=batch_size,
                                                         trace_length=trace_length,
                                                         target_model=target_model,
                                                         optimization_steps=optimistion_steps)
                print("update Target network")
                target_model.soft_update_from(model)
                model.log(encoder_summary=ae_summary, drqn_summary=drqn_summary)

            if n % 50:
                print("Saving")
                model.save(n)

            epsilon = max(epsilon * epsilon_decay, epsilon_min)

            #   epsilon decay
            losses.append(mean_loss)
            success_rate.append(successes / num_episodes)
            print("\repisode", n + 1, "success rate", success_rate[-1], 'loss %.2f:' % losses[-1],
                  'exploration', epsilon, end=' ' * 10)

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
