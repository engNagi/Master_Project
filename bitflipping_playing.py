from DQnetwork_bitflipping import DQN
import tensorflow as tf
from bitflipping import BitFlip
import time

size = 50
env = BitFlip(reward_type="sparse", n=size)


target_model = DQN(action_n=size, fcl_dims=256, scope="target_model")

with tf.Session() as sess:
    target_model.set_session(sess)
    sess.run(tf.global_variables_initializer())
    target_model.load()

    for n in range(5):
        state, goal = env.reset()  # reset environment
        for t in range(size):
            action = target_model.sample_action(state, goal, 0)
            #   Order of variables returned form take_action method
            #   frame, agent_position, done, reward, obj_agent_dis, visible
            next_state, reward, done = env.step(action=action)

            state = next_state
            env.render()
            time.sleep(0.5)
            if done:
                break
        print("Success :", done)
