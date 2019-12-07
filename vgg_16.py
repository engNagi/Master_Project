import tensorflow as tf
import numpy as np
import glob, random, os
from tensorflow.python.ops.variables import Variable


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_path = "saved_models/"
model_name = model_path + 'model'


class Network(object):
    # Create model
    def __init__(self):
        self.image = tf.placeholder(tf.float32, [None, 300, 300, 3], name='image')
        self.resized_image = tf.image.resize_images(self.image, [256, 256])
        tf.summary.image('resized_image', self.resized_image, 20)

        self.z_mu = self.encoder(self.resized_image)
        self.reconstructions = self.decoder(self.z_mu)
        tf.summary.image('reconstructions', self.reconstructions, 20)


        self.loss = self.compute_loss()
        tf.summary.scalar('loss', self.loss)

        self.merged = tf.summary.merge_all()


    def encoder(self, x):
        conv_1_1 = tf.layers.conv2d(x, filters=4, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.relu)
        conv_1_2 = tf.layers.conv2d(conv_1_1, filters=4, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.relu)
        maxpool_1 = tf.layers.max_pooling2d(conv_1_2, pool_size=3, strides=2, padding="same")


        conv_2_1 = tf.layers.conv2d(maxpool_1, filters=8, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.relu)
        conv_2_2 = tf.layers.conv2d(conv_2_1, filters=8, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.relu)
        maxpool_2 = tf.layers.max_pooling2d(conv_2_2, pool_size=3, strides=2, padding="same")


        conv_3_1 = tf.layers.conv2d(maxpool_2, filters=16, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.relu)
        conv_3_2 = tf.layers.conv2d(conv_3_1, filters=16, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.relu)
        maxpool_3 = tf.layers.max_pooling2d(conv_3_2, pool_size=3, strides=2, padding="same")


        conv_4_1 = tf.layers.conv2d(maxpool_3, filters=32, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.relu)
        conv_4_1 = tf.layers.conv2d(maxpool_3, filters=32, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.relu)
        maxpool_4 = tf.layers.max_pooling2d(conv_4_1, pool_size=3, strides=2, padding="same")


        conv_5_1 = tf.layers.conv2d(maxpool_4, filters=64, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.relu)
        conv_5_2 = tf.layers.conv2d(conv_5_1, filters=64, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.relu)
        maxpool_5 = tf.layers.max_pooling2d(conv_5_2, pool_size=3, strides=2, padding="same")


        conv_6_1 = tf.layers.conv2d(maxpool_5, filters=128, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.relu)
        conv_6_2 = tf.layers.conv2d(conv_6_1, filters=128, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.relu)
        maxpool_6 = tf.layers.max_pooling2d(conv_6_2, pool_size=3, strides=2, padding="same")


        conv_7_1 = tf.layers.conv2d(maxpool_6, filters=256, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.relu)
        conv_7_2 = tf.layers.conv2d(conv_7_1, filters=256, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.relu)



        x = tf.layers.flatten(conv_7_2)
        z_mu = tf.layers.dense(x, units=1024, name='z_mu')
        #z_logvar = tf.layers.dense(x, units=32, name='z_logvar')
        return z_mu #z_logvar

    def decoder(self, z):
        bn = tf.layers.dense(z, 1024, activation=tf.nn.relu) # 65536
        bn_rsh = tf.reshape(bn, [-1, 8, 8, 16])
        x = tf.layers.conv2d(bn_rsh, )
        x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        #x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        #x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=3, strides=2, padding='same', activation=None)
        return x


    def compute_loss(self):
        logits_flat = tf.reshape(self.reconstructions, [32, -1])
        labels_flat = tf.reshape(self.resized_image, [32, -1])

        reconstruction_loss = tf.reduce_sum(tf.square(logits_flat - labels_flat), axis=1)
        vae_loss = tf.reduce_mean(reconstruction_loss)

        return vae_loss


def data_iterator(batch_size):
    data_files = glob.glob('/home/WIN-UNI-DUE/sjmonagi/Desktop/dataset/Train/**/VAE_FloorPlan*', recursive=True)
    while True:
        data = np.load(random.sample(data_files, 1)[0])
        np.random.shuffle(data)
        np.random.shuffle(data)
        N = data.shape[0]
        start = np.random.randint(0, N - batch_size)
        yield data[start:start + batch_size]


def train_vae():
    sess = tf.InteractiveSession()

    global_step = tf.Variable(0, name='global_step', trainable=False)

    writer = tf.summary.FileWriter('logdir_autoencoder',  sess.graph)

    network = Network()
    train_op = tf.train.AdamOptimizer(1e-3).minimize(network.loss, global_step=global_step)
    tf.global_variables_initializer().run()

    saver = tf.train.Saver(max_to_keep=1)
    step = global_step.eval()
    training_data = data_iterator(batch_size=32)

    try:
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        print("Model restored from: {}".format(model_path))
    except:
        print("Could not restore saved model")

    while True:
            images = next(training_data)
            x, loss_value, summary = sess.run([train_op, network.loss, network.merged],
                                              feed_dict={network.image: images})
            writer.add_summary(summary, step)

            if np.isnan(loss_value):
                raise ValueError('Loss value is NaN')
            if step % 10 == 0 and step > 0:
                print('step {}: training loss {:.6f}'.format(step, loss_value))
                save_path = saver.save(sess, model_name, global_step=global_step)
            if loss_value <= 35:
                print('step {}: training loss {:.6f}'.format(step, loss_value))
                save_path = saver.save(sess, model_name, global_step=global_step)
                break
            step += 1

def load_vae():
    graph = tf.Graph()
    with graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=graph)

        network = Network()
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=1)
        training_data = data_iterator(batch_size=32)

        try:
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
        except:
            raise ImportError("Could not restore saved model")

        return sess, network

if __name__ == "__main__":
    train_vae()
