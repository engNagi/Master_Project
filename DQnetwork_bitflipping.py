import random
import numpy as np
import tensorflow as tf
from collections import deque

class DQN:
    def __init__(self, action_n, scope,
                 im_height=180,
                 im_width=160,
                 fcl_dims=256,
                 buffer_size=50000,
                 save_path='DQN_model/atarix.ckpt'):

        self.action_n = action_n
        self.scope = scope
        self.save_path = save_path
        self.im_height = im_height
        self.im_width = im_width
        self.fc1_dims = fcl_dims
        self.buffer = deque()

        with tf.variable_scope(scope):
            # inputs and targets
            # self.inputs = tf.placeholder(tf.float32, shape=(None, 4, self.im_height, self.im_width),
            #                              name='inputs')

            # tensorflow convolution needs the order to be:
            # (num_samples, height, width, "color")
            # so we need to tranpose later
            self.inputs = tf.placeholder(tf.float32, shape=(None, 30), name="Inputs")
            self.goals_ = tf.placeholder(tf.float32, shape=(None, 30), name="Goals_")
            self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
            self.goals = tf.placeholder(tf.float32, shape=(None,), name='Goals')

            # calculate output and cost
            # # convolutional layers
            # Z = self.inputs / 255.0
            # Z = tf.transpose(Z, [0, 2, 3, 1])
            # conv1 = tf.contrib.layers.conv2d(Z, 32, 8, 4, activation_fn=tf.nn.relu)
            #
            # conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
            #
            # conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)
            #
            # # fully connected layers
            # flat = tf.contrib.layers.flatten(conv3)
            # dense1 =tf.contrib.layers.fully_connected(self.state_goals, self.fc1_dims,
            #                          activation=tf.nn.relu,
            #                          kernel_initializer=tf.variance_scaling_initializer(scale=2))
            state_goals = tf.concat((self.inputs, self.goals_), axis=1)
            dense1 = tf.contrib.layers.fully_connected(state_goals, self.fc1_dims, tf.nn.relu)

            # final output layer
            self.predict_op = tf.contrib.layers.fully_connected(dense1, action_n)

            selected_action_values = tf.reduce_sum(self.predict_op * tf.one_hot(self.actions, self.action_n),
                                                   reduction_indices=[1])

            self.cost = tf.reduce_mean(tf.square(self.goals - selected_action_values))
            self.train_op = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    def copy_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)

        ops = []
        for p, q in zip(mine, theirs):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)

        self.session.run(ops)

    def set_session(self, session):
        self.session = session

    def predict(self, states, goals_):
        return self.session.run(self.predict_op, feed_dict={self.inputs: states,
                                                            self.goals_: goals_})

    def update(self, states, actions, targets, goals_):
        c, _ = self.session.run(
            [self.cost, self.train_op],
            feed_dict={
                self.inputs: states,
                self.goals: targets,
                self.actions: actions,
                self.goals_: goals_
            }
        )
        return c

    def sample_action(self, state, goals_, eps):
        """Implements epsilon greedy algorithm"""
        if np.random.random() < eps:
            return np.random.choice(self.action_n)
        else:
            return np.argmax(self.predict([state], [goals_])[0])

    def load(self):
        self.saver = tf.train.Saver(tf.global_variables())
        load_was_success = True
        try:
            save_dir = '/'.join(self.save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            self.saver.restore(self.session, load_path)
        except:
            print("no saved model to load. starting new session")
            load_was_success = False
        else:
            print("loaded model: {}".format(load_path))
            saver = tf.train.Saver(tf.global_variables())
            episode_number = int(load_path.split('-')[-1])

    def save(self, n):
        self.saver.save(self.session, self.save_path, global_step=n)
        print("SAVED MODEL #{}".format(n))

    def calculate_targets(self, next_states, dones, rewards, goals_, gamma):
        # Calculate targets
        next_Qs = self.predict(next_states, goals_)
        next_Q = np.amax(next_Qs, axis=1)
        targets = rewards + np.invert(dones).astype(np.float32) * gamma * next_Q

        return targets

    def remember(self, ep_experience):
        self.buffer.extend(ep_experience)


    def optimize(self, model, target_model, buffer, optimization_steps, batch_size, gamma):
        losses = 0

        for _ in range(optimization_steps):
            if len(self.buffer) < batch_size:  # if there's no enough transitions, do nothing
                return 0
            # sample batches from experiences
            else:
                samples = random.sample(self.buffer, batch_size)
                states, actions, rewards, next_states, dones, goals_ = map(np.array, zip(*samples))

            # Calculate targets
            next_Qs = target_model.predict(next_states, goals_)
            next_Q = np.amax(next_Qs, axis=1)
            targets = rewards + np.invert(dones).astype(np.float32) * gamma * next_Q

            #   Calculate network loss
            loss = model.update(states, actions, targets, goals_)
            losses += loss
        return losses / optimization_steps

    # def remember(self, ep_experience):
    #     self.memory += ep_experience.memory
    #     if len(self.memory) > self.buffer_size:
    #         self.memory = self.memory[-self.buffer_size:]
