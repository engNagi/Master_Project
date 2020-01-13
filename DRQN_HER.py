import random
import numpy as np
import tensorflow as tf
from collections import deque

random.seed(123)
np.random.seed(123)


class DRQN(object):
    def __init__(self, action_n, scope,
                 im_height=180,
                 im_width=160,
                 fcl_dims=256,
                 buffer_size=50000,
                 gamma=0.98,
                 nodes_num=518,
                 save_path='/home/nagi/Desktop/Master_Project/DRQN/DRQN.ckpt'):

        self.action_n = action_n
        self.scope = scope
        self.save_path = save_path
        self.im_height = im_height
        self.im_width = im_width
        self.fc1_dims = fcl_dims
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.nodes_num = nodes_num
        self.buffer = deque(maxlen=self.buffer_size)

        with tf.variable_scope(scope):
            # seperate agent observation, and positions
            self.inputs = tf.placeholder(tf.float32, shape=(None, 512), name="features_positions")
            # additional goals
            self.goals = tf.placeholder(tf.float32, shape=(None, 512), name="Goals_")
            #   actions
            self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
            # Q-targets-values
            self.Q_values = tf.placeholder(tf.float32, shape=(None,), name='Targets_Q_Values')
            # concatenated agent position and observation with the goal_position
            state_goals = tf.concat((self.inputs, self.goals), axis=1)

            with tf.variable_scope("RNN"):
                self.train_length = tf.placeholder(tf.int32)
                self.batch_size = tf.placeholder(tf.int32, shape=[])
                self.input_flat = tf.reshape(tf.layers.flatten(state_goals),
                                             [self.batch_size, self.train_length, 1024])

                # number_of_units may need to be changed
                self.cell = tf.contrib.rnn.LSTMCell(num_units=self.nodes_num,
                                                    state_is_tuple=True,
                                                    activation=tf.nn.tanh,
                                                    initializer=tf.initializers.he_normal())

                self.state_in = self.cell.zero_state(self.batch_size, tf.float32)
                self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.input_flat,
                                                             cell=self.cell,
                                                             dtype=tf.float32,
                                                             initial_state=self.state_in,
                                                             scope=scope + '_rnn')

                self.rnn_flat = tf.reshape(self.rnn, shape=[-1, self.nodes_num])

            dense1 = tf.layers.dense(self.rnn_flat, self.fc1_dims, activation=tf.nn.relu, trainable=True)

            # final output layer
            self.predict_op = tf.layers.dense(dense1, self.action_n, trainable=True)

            actions_q_values = tf.reduce_sum(self.predict_op * tf.one_hot(self.actions, self.action_n),
                                             reduction_indices=[1])

            self.clipped_Q_values = tf.clip_by_value(self.Q_values, -1 / (1 - self.gamma), 0)

            self.cost = tf.reduce_mean(tf.square(self.clipped_Q_values - actions_q_values))

            self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.cost)

            tf.summary.scalar("Cost", self.cost)
            tf.summary.histogram("Goals", self.goals)
            tf.summary.histogram("Action_Q_values", self.Q_values)
            tf.summary.histogram("LSTM", self.rnn)
            tf.summary.histogram("LSTM_State", self.rnn_state)
            self.merged = tf.summary.merge_all()

    def hard_update_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)

        self.session.run([v_t.assign(v) for v_t, v in zip(mine, theirs)])

    def soft_update_from(self, other, tau=0.95):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)

        self.session.run([v_t.assign(v_t * (1. - tau) + v * tau) for v_t, v in zip(mine, theirs)])

    def set_session(self, session):
        self.session = session

    def predict(self, pos_obs_state, goals, batch_size, trace_length, rnn_state):
        actions_q_values, rnn, rnn_state_ = self.session.run([self.predict_op, self.rnn, self.rnn_state],
                                                             feed_dict={self.goals: goals,
                                                                        self.state_in: rnn_state,
                                                                        self.inputs: pos_obs_state,
                                                                        self.batch_size: batch_size,
                                                                        self.train_length: trace_length})
        return actions_q_values, rnn, rnn_state_

    def update(self, goals, states, actions, batch_size, q_values, trace_length, rnn_state):
        self.c, _, self.summary = self.session.run([self.cost, self.train_op, self.merged],
                                                   feed_dict={self.goals: goals,
                                                              self.inputs: states,
                                                              self.actions: actions,
                                                              self.Q_values: q_values,
                                                              self.state_in: rnn_state,
                                                              self.batch_size: batch_size,
                                                              self.train_length: trace_length})
        return self.c, self.summary

    def sample_action(self, goal, batch_size, trace_length, epsilon, rnn_state, features):
        """Implements epsilon greedy algorithm"""
        if np.random.random() < epsilon:
            q_values, rnn, rnn_state_ = self.predict([features], [goal], batch_size, trace_length, rnn_state)
            action = np.random.choice(self.action_n)
        else:
            action_q_values, _, rnn_state_ = self.predict([features], [goal], batch_size, trace_length, rnn_state)
            action = np.argmax(action_q_values[0])
        return action, rnn_state_

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

    def rnn_sample(self, batch_size, trace_length):
        sampled_traces = []
        for i in range(batch_size):
            idx = np.random.randint(0, len(self.buffer))
            if idx + trace_length > len(self.buffer):
                idx -= trace_length
            sampled_traces.append(list(self.buffer)[idx:idx + trace_length])
        sampled_traces = np.array(sampled_traces)
        sampled_traces = np.reshape(sampled_traces, [batch_size * trace_length, 6])
        return sampled_traces

    def optimize(self, model, target_model, batch_size, train_batch, trace_length):
        losses = 0
        rnn_stat_train = (np.zeros([batch_size, self.nodes_num]), np.zeros([batch_size, self.nodes_num]))
        # if len(self.buffer) < batch_size:  # if there's no enough transitions, do nothing
        #     return 0
        # # sample batches from experiences
        # else:
        # samples = self.rnn_sample(batch_size, trace_length)
        states, actions, rewards, next_states, dones, goals = map(np.array, zip(*train_batch))
        # Calculate targets
        states = np.vstack(train_batch[:, 0])
        actions = train_batch[:, 1]
        rewards = train_batch[:, 2]
        next_states = np.vstack(train_batch[:, 3])
        goals = np.vstack(train_batch[:, 5])
        dones = train_batch[:, 4]
        next_Qs, _, _ = target_model.predict(goals=goals,
                                             batch_size=batch_size,
                                             pos_obs_state=next_states,
                                             trace_length=trace_length,
                                             rnn_state=rnn_stat_train)
        next_Q = np.amax(next_Qs, axis=1)
        target_q_values = rewards + np.invert(dones).astype(np.float32) * self.gamma * next_Q
        #   Calculate network loss
        loss, summary = model.update(goals=goals,
                                  states=states,
                                  actions=actions,
                                  batch_size=batch_size,
                                  q_values=target_q_values,
                                  trace_length=trace_length,
                                  rnn_state=rnn_stat_train)
        return loss, summary

    def log(self, drqn_summary, total_steps):

        # encoder_writer = tf.summary.FileWriter("/home/nagi/Desktop/Master_Project/DRQN/encoder")
        # encoder_writer.add_summary(encoder_summary, global_step=total_steps)
        writer = tf.summary.FileWriter("/home/nagi/Desktop/Master_Project/DRQN/Train")
        writer.add_summary(drqn_summary, global_step=total_steps)
        #aux_writer = tf.summary.FileWriter("/home/nagi/Desktop/Master_Project/DRQN/aux")

        # aux_summary = tf.Summary()
        # aux_summary.value.add(tag="success_rate", simple_value=success_rate)
        # aux_summary.value.add(tag="failure_rate", simple_value=failure_rate)
        # aux_summary.value.add(tag="ratio", simple_value=success_failure_ratio)
        # aux_writer.add_summary(aux_summary, success_rate, global_step=total_steps)
        # aux_writer.add_summary(aux_summary, failure_rate, total_steps)
        # aux_writer.add_summary(aux_summary, success_failure_ratio)

# def log_rnn(self):
#     writer = tf.summary.FileWriter("/home/nagi/Desktop/Master_Project/DRQN/Train")
#     writer.add_summary(self.summary)
