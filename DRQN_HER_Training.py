import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly_express as px
import plotly.io as pio
import cufflinks as cf
import matplotlib.pyplot as plt
cf.go_offline()
pio.renderers.default = "browser"
from autoencoder import load_autoencoder
from DRQN_HER import DRQN
from Her_episodes_experiences import Her_rec_experiences
from experience_buffer import experience_buffer
from Environment_dataset_generation import Environment

random.seed(123)
np.random.seed(123)

action_n = 6
#########################   hyper-parameter
num_episodes = 50500
#max_episode_length = 150
her_strategy = "future"
her_samples = 4
batch_size = 16
trace_length = 8

# DQN  parameters
gamma = 0.99
fcl_dims = 512
nodes_num = 256
# optimistion_steps = 200
update_period = 5
pretrain_steps = 1000

#   tracking parameters
data_plots = pd.DataFrame(columns=["Episode", "Successes", "Failures", "Ratio"])
loss = 0

#   environment initialization
env = Environment(random_goals=True, random_init=True)
env.make()

#   Autoenconder
print("Autoencoder")
ae_sess, ae = load_autoencoder()

# epsilon for Epsilon Greedy Algorithm
epsilon = 1
epsilon_min = 0.02
epsilon_decay = 0.999

# experience replay parameters
her_rec_buffer = Her_rec_experiences()

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

        global_step = tf.Variable(0, name="global_step", trainable=False)

        tf.global_variables_initializer().run()
        model.load()

        start = global_step.eval(session=drqn_sess)

        successes = 0
        failures = 0
        total_steps = 0
        for n in range(start, num_episodes):

            # episode buffer
            episode_buffer = experience_buffer()

            #   rnn_init_state
            rnn_state = (np.zeros([1, nodes_num]), np.zeros([1, nodes_num]))

            # reset environment
            obs_state, pos_state, goal, distance, _, _ = env.reset()

            features = ae_sess.run(ae.feature_vector, feed_dict={ae.image: obs_state[None, :, :, :]})
            features = np.squeeze(features, axis=0)

            done = False
            while not done:

                action, rnn_state_ = model.sample_action(goal=goal,
                                                         batch_size=1,
                                                         trace_length=1,
                                                         epsilon=epsilon,
                                                         rnn_state=rnn_state,
                                                         features=features)

                obs_state_, pos_state_, distance_, done, reward, collision = env.step(action, goal, distance)

                features_, ae_summary = ae_sess.run([ae.feature_vector, ae.merged],
                                                    feed_dict={ae.image: obs_state_[None, :, :, :]})
                features_ = np.squeeze(features_, axis=0)

                episode_buffer.add(
                    np.reshape(np.array([features, pos_state, action, reward, features_, pos_state_, done, goal]),
                               [1, 8]))

                if total_steps > pretrain_steps:

                    if total_steps % (update_period * 1000) == 0:
                        print("update Target network")
                        target_model.soft_update_from(model)

                    if total_steps % (update_period * 100) == 0:
                        her_rec_buffer.her(strategy=her_strategy, her_samples=her_samples)

                        train_batch = her_rec_buffer.sample(batch_size=batch_size, trace_length=trace_length)

                        loss, drqn_summary = model.optimize(model=model,
                                                            batch_size=batch_size,
                                                            train_batch=train_batch,
                                                            trace_length=trace_length,
                                                            target_model=target_model)
                        model.log(drqn_summary=drqn_summary, encoder_summary=ae_summary, step=start)

                total_steps += 1
                rnn_state = rnn_state_
                features = features_
                distance = distance_
                pos_state = pos_state_

                if done:
                    if total_steps > pretrain_steps:
                        if distance == 0:
                            successes += done
                        else:
                            failures += done
                    break
            her_rec_buffer.add(episode_buffer.memory)
            epsilon = max(epsilon * epsilon_decay, epsilon_min)

            print("\repisode:", n + 1, "successes:", successes, "goal x:%.2f" % goal[0], "goal z:%.2f" % goal[2]
                  , "distance: %3f" % distance, "failures", failures, "ratio %.3f" % (successes / (failures + 1e-6)),
                  "loss: %.2f" % loss, "exploration %.2f" % epsilon)

            data_plots = data_plots.append({"Episodes": str(n), "Successful trajectories": successes,
                                            "Failed trajectories": failures, "Ratio": (successes / (failures + 1e-6))},
                                           ignore_index=True)

            loss_plots = data_plots.append({"Episodes": str(n), "Loss": loss}, ignore_index=True)

            if n % 1000 == 0 and n > 0:
                # combined plot of successful, failed trajectories Ratio between them
                data_plots.plot(x="Episodes", y=["Successful trajectories", "Failed trajectories", "Ratio"],
                                title="Agent Learning Ratio")
                plt.xlabel("Episodes")
                plt.ylabel("Successful/Failed Trajectories and Ratio")
                plt.show()
                fig = plt.gcf()
                fig.savefig("plot_failed_success_ratio.png")

                # plot of successful trajectories
                data_plots.plot(x="Episodes", y=["Successful trajectories"],
                                title="Successful trajectories")
                plt.xlabel("Episodes")
                plt.ylabel("Successful trajectories")
                plt.show()
                fig = plt.gcf()
                fig.savefig("Successful trajectories.png")

                # plot of Failed trajectories
                data_plots.plot(x="Episodes", y=["Failed trajectories"],
                                title="Failed trajectories")
                plt.xlabel("Episodes")
                plt.ylabel("Failed trajectories")
                plt.show()
                fig = plt.gcf()
                fig.savefig("Failed trajectories.png")

                # plot of Ratio between Successful and failed trajectories
                data_plots.plot(x="Episodes", y=["Ratio"],
                                title="Ratio between successful and Failed Trajectories")
                plt.xlabel("Episodes")
                plt.ylabel("Ratio")
                plt.show()
                fig = plt.gcf()
                fig.savefig("Ratio.png")

                loss_plots.plot(x="Episodes", y=["Loss"], title="HER-DRQN model loss")
                plt.xlabel("Episodes")
                plt.ylabel("Loss")
                plt.show()
                fig = plt.gcf()
                fig.savefig("HER-DRQN model loss.png")

            #   saving
            if n % 500 == 0 and n > 0:
                global_step.assign(n).eval()
                model.save(global_step, n)


data_plots.plot(x="Episodes", y=["Successful trajectories", "Failed trajectories", "Ratio"],
                title="Agent Learning Ratio")
plt.xlabel("Episodes")
plt.ylabel("Successful/Failed Trajectories and Ratio")
plt.show()
fig = plt.gcf()
fig.savefig("plot_failed_success_ratio.png")

# plot of successful trajectories
data_plots.plot(x="Episodes", y=["Successful trajectories"],
                title="Successful trajectories")
plt.xlabel("Episodes")
plt.ylabel("Successful trajectories")
plt.show()
fig = plt.gcf()
fig.savefig("Successful trajectories.png")

# plot of Failed trajectories
data_plots.plot(x="Episodes", y=["Failed trajectories"],
                title="Failed trajectories")
plt.xlabel("Episodes")
plt.ylabel("Failed trajectories")
plt.show()
fig = plt.gcf()
fig.savefig("Failed trajectories.png")

# plot of Ratio between Successful and failed trajectories
data_plots.plot(x="Episodes", y=["Ratio"],
                title="Ratio between successful and Failed Trajectories")
plt.xlabel("Episodes")
plt.ylabel("Ratio")
plt.show()
fig = plt.gcf()
fig.savefig("Ratio.png")

loss_plots.plot(x="Episodes", y=["Loss"], title="HER-DRQN model loss")
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.show()
fig = plt.gcf()
fig.savefig("HER-DRQN model loss.png")
