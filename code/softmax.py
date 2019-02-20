import tensorflow as tf
from prepare_data import *
import matplotlib.pyplot as plt
import time
from datetime import timedelta
from stats import calc_f_measure

MODEL_SOFTMAX_FNAME = TRAINED_MODELS + 'softmax.ckpt'
MODEL_SOFTMAX_FNAME_META = TRAINED_MODELS + 'softmax.ckpt.meta'
SOFTMAX_ITERATIONS = 5000
SOFTMAX_GRAD_DEC_ALPHA = 0.01
SOFTMAX_TRAIN_PERCENT = 0.80

log_name = "logs/softmax.txt"

def main_softmax(iter_num = SOFTMAX_ITERATIONS):
    categories = 7

    landmarks, emos = get_all_landmarks_emos_flatten()
    features = len(landmarks[0])

    x = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, categories])
    W = tf.Variable(tf.zeros([features, categories]), name = "W")
    b = tf.Variable(tf.zeros([categories]), name="b")

    # Softmax
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    loss = -tf.reduce_mean(y_ * tf.log(y))
    update = tf.train.GradientDescentOptimizer(SOFTMAX_GRAD_DEC_ALPHA).minimize(loss)

    #data_x = np.array([[2,4],[3,9],[4,16],[6,36],[7,49]]) - 2 features
    #data_y = np.array([[0, 1, 0],[0, 1, 0],[1, 0, 0],[1, 0, 0],[0, 1, 0]]) - for 3 labels

    # Split to train/test
    rnd_indices = np.random.rand(len(landmarks)) < SOFTMAX_TRAIN_PERCENT
    train_x = np.array(landmarks)[rnd_indices]
    train_y = np.array(emos)[rnd_indices]
    test_x = np.array(landmarks)[~rnd_indices]
    test_y = np.array(emos)[~rnd_indices]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver = tf.train.Saver([W, b])

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train the model
    time_start = time.time()
    for i in range(iter_num):
        sess.run(update, feed_dict = {x:train_x, y_:train_y})
        if i % 1000 == 0 :
            last_loss = loss.eval(session=sess, feed_dict = {x:train_x, y_:train_y})
            print('Iteration:' , i , ' b:' , sess.run(b), ' loss:', last_loss)
        time_end = time.time()

    # Save the trained model
    save_path = saver.save(sess, MODEL_SOFTMAX_FNAME)
    print("Model saved in file: %s, processing time: %s" % (save_path, str(timedelta(seconds=time_end - time_start))))

    # Test the data
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))

    # Test & draw
    pred_y = sess.run(y, feed_dict={x: test_x})

    write_log(log_name, "Softmax, %s" % str(datetime.now()))
    write_log(log_name, "Iterations: %d" % iter_num)
    write_log(log_name, "Accuracy: %s, last_loss: %s" % (sess.run(accuracy, feed_dict={x: test_x, y_: test_y}), last_loss))
    write_log(log_name, "Eval time: %s" % str(timedelta(seconds=time_end - time_start)))

    fig, ax = plt.subplots()
    ax.scatter(test_y, pred_y)
    ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

    # calc stats: true positive etc
    stats = calc_f_measure(test_y, pred_y)
    write_log(log_name, "Stats: %s" % stats)

    print("Sasha")
    sess.close()

def init_softmax_pretrained():
    # Init softmax - see evaluate_softmax
    tf.reset_default_graph()

    sess = tf.Session()
    saver = tf.train.import_meta_graph(MODEL_SOFTMAX_FNAME_META)
    saver.restore(sess, tf.train.latest_checkpoint(TRAINED_MODELS))

    b = sess.run('b:0')
    W = sess.run('W:0')

    sess.close()

    return {"W": W, "b": b}

def predict_softmax(W, b, landmark):
    # Softmax - the result is vector
    rc = np.matmul(np.array(landmark), W) + b
    mask = rc < 0
    rc[mask] = 0.0
    return rc / rc.max()  # return vector with values between 0.0 to 1.0

def predict_softmax_(params, landmark):
    return predict_softmax(params["W"], params["b"], landmark)

# To run:
# i = 3333
# landmarks, emos = get_all_landmarks_emos_flatten()
# print(emos[i], evaluate_softmax(landmarks[i]))
def evaluate_softmax(landmark):
    W, b = init_softmax_pretrained()
    return predict_softmax(W, b, landmark)

def tune_softmax():
    for iter_num in [300000, 400000, 500000]:
        main_softmax(iter_num)

# till iters = 300k alpha = 0.001
# from iters 300k alpha = 0.01