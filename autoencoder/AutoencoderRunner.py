import numpy as np

import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# from autoencoder.autoencoder_models.Autoencoder import Autoencoder
from autoencoder_models.Autoencoder import Autoencoder

parameter_servers = ["192.168.122.101:2223"]
workers = ["192.168.122.101:2222",
           "192.168.122.102:2222"]
cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":

    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
        autoencoder = Autoencoder(n_input=784,
                                  n_hidden=200,
                                  transfer_function=tf.nn.softplus,
                                  optimizer=tf.train.AdamOptimizer(learning_rate=0.001))

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             init_op=autoencoder.init_op)

    with sv.prepare_or_wait_for_session(server.target) as sess:
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = get_random_block_from_data(X_train, batch_size)

                # Fit training using batch data
                cost = autoencoder.partial_fit(batch_xs, sess)
                # Compute average loss
                avg_cost += cost / n_samples * batch_size

                # Display logs per epoch step
                if epoch % display_step == 0:
                    print "Epoch:", '%04d' % (epoch + 1), \
                        "cost=", "{:.9f}".format(avg_cost)

        print "Total cost: " + str(autoencoder.calc_total_cost(X_test, sess))
