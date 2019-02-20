import tensorflow as tf
from prepare_data import *
import matplotlib.pyplot as plt
import time
from datetime import timedelta, datetime
from stats import calc_f_measure

TRAINED_MODELS_SINGLE_PERCEPTRON = 'trained_models/perceptron_single/'
TRAINED_MODELS_MULT_PERCEPTRON = 'trained_models/perceptron_mult/'

MODEL_SINGLE_PERCEPTRON = TRAINED_MODELS_SINGLE_PERCEPTRON + 'single_layer_perceptron.ckpt'
MODEL_MULT_PERCEPTRON = TRAINED_MODELS_MULT_PERCEPTRON + 'mult_layers_perceptron.ckpt'

MODEL_SINGLE_PERCEPTRON_FNAME_META = TRAINED_MODELS_SINGLE_PERCEPTRON + 'single_layer_perceptron.ckpt.meta'
MODEL_MULT_PERCEPTRON_FNAME_META = TRAINED_MODELS_MULT_PERCEPTRON + 'mult_layers_perceptron.ckpt.meta'

DEFAULT_ITERATIONS_NUMBER = 350000

log_name = "logs/perceptron.txt"

def get_next_batch(bnum, bsize, xs, ys):
    bsize = bsize % len(xs) # len(xs) = len(ys)
    bnum = bnum % (int(len(xs)/bsize))   # 2do: this is not precise

    xs_from_end = xs[bnum*bsize:(bnum+1)*bsize]
    if len(xs_from_end) < bsize:
        batch_xs = xs_from_end + xs[:bsize-len(xs_from_end)]
    else:
        batch_xs = xs_from_end

    ys_from_end = ys[bnum*bsize:(bnum+1)*bsize]
    if len(ys_from_end) < bsize:
        batch_ys = ys_from_end + ys[:bsize-len(ys_from_end)]
    else:
        batch_ys = ys_from_end

    return batch_xs, batch_ys

def perceptron_single_layer(iter_num = DEFAULT_ITERATIONS_NUMBER):
    categories = 7

    landmarks, emos = get_all_landmarks_emos_flatten()
    features = len(landmarks[0])

    x = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, categories])

    W = tf.Variable(tf.zeros([features, categories]), name = "W")
    b = tf.Variable(tf.zeros([categories]), name="b")

    # Softmax
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    #data_x = np.array([[2,4],[3,9],[4,16],[6,36],[7,49]]) - 2 features
    #data_y = np.array([[0, 1, 0],[0, 1, 0],[1, 0, 0],[1, 0, 0],[0, 1, 0]]) - for 3 labels

    # Split to train/test
    rnd_indices = np.random.rand(len(landmarks)) < 0.80
    train_x = np.array(landmarks)[rnd_indices]
    train_y = np.array(emos)[rnd_indices]
    test_x = np.array(landmarks)[~rnd_indices]
    test_y = np.array(emos)[~rnd_indices]

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    time_start = time.time()
    for i in range(iter_num):
        batch_xs, batch_ys = get_next_batch(i, 100, train_x, train_y)  # MB-GD
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    time_end = time.time()

    # Save the trained model
    save_path = saver.save(sess, MODEL_SINGLE_PERCEPTRON)
    print("Model saved in file: %s" % save_path)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    write_log(log_name, "Single Perceptron, %s" % str(datetime.now()))
    write_log(log_name, "Iterations: %d" % iter_num)
    write_log(log_name,"Accuracy: %s" % sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
    write_log(log_name,"Eval time: %s" % str(timedelta(seconds=time_end - time_start)))

    print_perceptron_stats(test_x, test_y, "single")

def perceptron_two_hidden_layers(iter_num = DEFAULT_ITERATIONS_NUMBER):
    l1 = 200
    l2 = 136
    (hidden1_size, hidden2_size) = (200, 136)

    categories = 7

    landmarks, emos = get_all_landmarks_emos_flatten()
    features = len(landmarks[0])

    x = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, categories])

    W1 = tf.Variable(tf.truncated_normal([features, hidden1_size], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[hidden1_size]))
    z1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    W2 = tf.Variable(tf.truncated_normal([hidden1_size, hidden2_size], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[hidden2_size]))
    z2 = tf.nn.relu(tf.matmul(z1, W2) + b2)

    W3 = tf.Variable(tf.truncated_normal([hidden2_size, categories], stddev=0.1), name="W")
    b3 = tf.Variable(tf.constant(0.1, shape=[categories]), name="b")

    y = tf.nn.softmax(tf.matmul(z2, W3) + b3)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

    # Split to train/test
    rnd_indices = np.random.rand(len(landmarks)) < 0.80
    train_x = np.array(landmarks)[rnd_indices]
    train_y = np.array(emos)[rnd_indices]
    test_x = np.array(landmarks)[~rnd_indices]
    test_y = np.array(emos)[~rnd_indices]

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    time_start = time.time()
    for i in range(iter_num):
        batch_xs, batch_ys =get_next_batch(i, 100, train_x, train_y)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    time_end = time.time()

    # Save the trained model
    save_path = saver.save(sess, MODEL_MULT_PERCEPTRON)
    print("Model saved in file: %s" % save_path)


    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    write_log(log_name, "Mult Perceptron (%d, %d), %s" % (l1, l2, str(datetime.now())))
    write_log(log_name, "Iterations: %d" % iter_num)
    write_log(log_name, "Accuracy: %s" % sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
    write_log(log_name, "Eval time: %s" % str(timedelta(seconds=time_end - time_start)))

    print_perceptron_stats(test_x, test_y, "mult")

def restore_W_b(layers):
    if layers == "single":
        meta_fname = MODEL_SINGLE_PERCEPTRON_FNAME_META
        dname = TRAINED_MODELS_SINGLE_PERCEPTRON
    else:
        meta_fname = MODEL_MULT_PERCEPTRON_FNAME_META
        dname = TRAINED_MODELS_MULT_PERCEPTRON

    tf.reset_default_graph()

    sess = tf.Session()
    saver = tf.train.import_meta_graph(meta_fname)
    saver.restore(sess, tf.train.latest_checkpoint(dname))

    b = sess.run('b:0')
    W = sess.run('W:0')

    sess.close()
    return W, b

# To run:
# i = 3333
# landmarks, emos = get_all_landmarks_emos_flatten()
# print(emos[i], evaluate_restored_perceptron_single_landmark(landmarks[i]), "single") or "mult"
def evaluate_restored_perceptron_single_landmark(landmark, layers="single"):
    W, b = restore_W_b(layers)
    # Softmax - the result is vector
    rc = np.matmul(np.array(landmark), W) + b
    mask = rc < 0
    rc[mask] = 0.0
    return rc/rc.max() # return vector with values between 0.0 to 1.0

def print_perceptron_stats(test_x, test_y, layers):
    W, b = restore_W_b(layers)
    perceptron_rc = []
    for i in range(len(test_x)):
        predicted_y = np.matmul(test_x[i], W) + b
        perceptron_rc.append(predicted_y)
    write_log(log_name, "Stats: %s" % calc_f_measure(test_y, perceptron_rc))

def tune_perceptron():
    # for iter_nums in [700, 1000]:
    #     perceptron_single_layer(iter_nums)
    for iter_nums in [300000]:
        perceptron_two_hidden_layers(iter_nums)