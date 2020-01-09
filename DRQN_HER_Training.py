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
num_epochs = 1000
num_episodes = 10000
max_episode_length = 150
her_strategy = "future"
her_samples = 4

# experience replay parameters
ep_experience = Her_episodes_experiences()

# DQN Bathrooms parameters
batch_size = 32
trace_length = 8
gamma = 0.99
fcl_dims = 512
nodes_num = 512
optimistion_steps = 200
update_period = 1000
pretrain_steps = 1000
#   tracking parameters
losses = []
loss = 0
loss_ = 0
success_rate = []
failure_rate = []
success_failure_ratio = []
episode_reward = 0
drqn_summary = tf.Summary()

#   environment initialization
env = Environment(random_goals=True, random_init=True)
env.make()

#   Autoenconder
print("Autoencoder")
ae_sess, ae = load_autoencoder()

# Create original and target  Networks


# epsilon for Epsilon Greedy Algorithm
epsilon = 1
epsilon_min = 0.02
epsilon_decay = 0.999

#   main loop
print("DQN_HER_Model")
drqn_graph = tf.Graph()
with drqn_graph.as_default():
    model = DRQN(action_n=action_n, nodes_num=nodes_num, fcl_dims=fcl_dims, scope="model")
    target_model = DRQN(action_n=action_n, nodes_num=nodes_num, fcl_dims=fcl_dims, scope="target_model")

drqn_sess = tf.Session(graph=drqn_graph)
model.set_session(drqn_sess)
target_model.set_session(drqn_sess)
with drqn_sess.as_default():
    with drqn_graph.as_default():
        tf.global_variables_initializer().run()
        model.load()

        # for j in range(num_epochs):
        successes = 0
        failures = 0
        total_steps = 0
        for n in range(num_episodes):
            # reset environment
            obs_state, pos_state, goal, obj_agent_dis, _, _ = env.reset()

            features = ae_sess.run(ae.feature_vector, feed_dict={ae.image: obs_state[None, :, :, :]})
            features = np.squeeze(features, axis=0)
            obs_pos_state = np.concatenate((features, pos_state), axis=0)

            rnn_state = (np.zeros([1, nodes_num]), np.zeros([1, nodes_num]))
            done = False
            while not done:

                action, rnn_state_ = model.sample_action(goal=goal,
                                                         batch_size=1,
                                                         trace_length=1,
                                                         epsilon=epsilon,
                                                         rnn_state=rnn_state,
                                                         obs_pos_state=obs_pos_state)

                obs_state_, pos_state_, done, reward, object_agent_dis_, visible, _, collided = env.step(action,
                                                                                                         obj_agent_dis)
                # if visible and collided:
                #     print(" \nstep:", i, "visibility:", visible, ", collide:", collided, end=' ' * 10)
                features_, ae_summary = ae_sess.run([ae.feature_vector, ae.merged],
                                                    feed_dict={ae.image: obs_state[None, :, :, :]})
                features_ = np.squeeze(features_, axis=0)
                obs_pos_state_ = np.concatenate((features_, pos_state_), axis=0)

                # append to experience replay
                ep_experience.add(obs_pos_state, action, reward, obs_pos_state_, done, goal)
                if visible and not collided:
                    successes += done
                else:
                    failures += done
                #success_failure_ratio.append((successes/(failure_rate+1e-6)))
                total_steps += 1
                if total_steps > pretrain_steps:
                    # HER
                    ep_memory = ep_experience.her(strategy="future", her_samples=her_samples)
                    model.buffer.extend(ep_memory)
                    if total_steps % update_period == 0:
                        print("update_main_network")
                        target_model.soft_update_from(model)
                    loss, drqn_summary = model.optimize(model=model,
                                                        batch_size=batch_size,
                                                        trace_length=trace_length,
                                                        target_model=target_model)

                rnn_state = rnn_state_
                obs_pos_state = obs_pos_state_
                obj_agent_dis = object_agent_dis_
                # model.log(encoder_summary=ae_summary,
                #           drqn_summary=drqn_summary,
                #           success_rate=successes,
                #           failure_rate=failures,
                #           success_failure_ratio=(successes / (failures+1e-6)))
                if done:
                    break


            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            losses.append(loss)
            success_rate.append(successes/(failures+1e-6))
            print("\repisode:", n + 1, "successes:", successes,
                  "failures", failures, "ratio %.3f" % (successes / (failures + 1e-6)),
                  'loss: %.2f' % loss, 'exploration %.2f' % epsilon)


            if n % 50 == 0 and n > 0:
                print("saving")
                model.save(n)

# Plots
plt.plot(loss)
plt.xlabel('episodes')
plt.ylabel('Losses')
plt.savefig("losses.png")
plt.show()

plt.plot(success_rate)
plt.xlabel('episodes')
plt.ylabel('Success rate')
plt.savefig("Success.png")
plt.show()

plt.plot(failure_rate)
plt.xlabel('episodes')
plt.ylabel('failure rate')
plt.savefig("failure.png")
plt.show()

plt.plot(success_failure_ratio)
plt.xlabel('episodes')
plt.ylabel('Success/failure ratio')
plt.savefig("Success_failure_ratio.png")
plt.show()
