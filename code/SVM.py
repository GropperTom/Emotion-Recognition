import tensorflow as tf
import numpy as np
from DataBase import SVM_DataBase

TENSORBOARD_DIR = "tensorboard\SVM"
#tensorboard --logdir=C:\Users\gropp\PycharmProjects\ML_Project\tensorboard\SVM

CLASSES_NUM = 7

class SVMModel:
    def __init__(self, db, neurons_num, learning_rate, alpha, dropout_percent=None):
        self.train_data = db.train["data"]
        self.train_labels = db.train["labels"]
        self.test_data = db.test["data"]
        self.test_labels = db.test["labels"]
        self.x_max_length = db.max_length
        self.graph = tf.Graph()
        self.session = None
        self.hidden_layers_num = db.get_hidden_layers_num()
        self.features_num = db.get_features_num()
        with self.graph.as_default():
            with tf.name_scope("SVM_model"):
                with tf.name_scope("inner_RNN_model"):
                    self.size_of_x = tf.placeholder(tf.int32, shape=())
                    self.x = tf.placeholder(tf.float32
                                            , [None, self.x_max_length, self.features_num]
                                            , name="data")
                    self.y = tf.placeholder(tf.float32
                                            , [None, 1]
                                            , name="labels")
                    with tf.name_scope("LSTM"):
                        with tf.name_scope("LSTM_cells"):
                            LSTM_cells = tf.nn.rnn_cell.BasicLSTMCell(neurons_num
                                                                      , activation=tf.nn.sigmoid)
                            if dropout_percent is not None:
                                LSTM_cells = tf.nn.rnn_cell.DropoutWrapper(LSTM_cells
                                                                           , output_keep_prob=dropout_percent)

                        non_zero_in_x = tf.sign(tf.reduce_max(tf.abs(self.x)
                                                              , 2))
                        x_lengths = tf.reduce_sum(non_zero_in_x
                                                  , 1)
                        x_lengths = tf.cast(x_lengths
                                            , tf.int32)
                        output, _ = tf.nn.dynamic_rnn(LSTM_cells
                                                      , self.x
                                                      , x_lengths
                                                      , dtype=tf.float32)
                        output = tf.nn.relu(output)

                        last = self.gather_relevant(output, x_lengths)

                weights = tf.Variable(tf.truncated_normal([neurons_num, 1]
                                                          , stddev=0.1)
                                      , name="weights")
                bias = tf.Variable((tf.ones([1]) * 0.1)
                                   , name="bias")
                self.prediction = tf.subtract(tf.matmul(last
                                                        , weights)
                                              , bias
                                              , name="prediction")
                l2_norm = tf.reduce_sum(tf.square(weights))
                alpha = tf.constant([alpha])
                classification_term = tf.reduce_mean(tf.maximum(0.
                                                                , tf.subtract(1.
                                                                              , tf.multiply(self.prediction
                                                                                            , self.y))))
                loss = tf.add(tf.multiply(alpha
                                          , l2_norm)
                              , classification_term)

                self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss
                                                                                , name="optimizer")

                tf.summary.histogram("weights_histogram"
                                     , weights)
                tf.summary.histogram("biases_histogram"
                                     , bias)
                tf.summary.scalar("loss_summary"
                                  , tf.squeeze(loss))

    def gather_relevant(self, output, lengths):
        index = tf.range(self.size_of_x) * self.x_max_length + (lengths - 1)
        relevant = tf.gather(tf.reshape(output, [-1, output.get_shape()[2]]), index)
        return relevant

    def run(self, iterations_num):
        with self.graph.as_default():
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(TENSORBOARD_DIR)
            writer.add_graph(self.graph)
            merged_summary = tf.summary.merge_all()

            test_samples_num = len([_ for _ in self.test_data])
            correct_prediction = tf.equal(self.y
                                          , tf.sign(self.prediction))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction
                                              , tf.float32)) * 100
            train_accuracy_summary = tf.summary.scalar("train_accuracy"
                                                       , accuracy)
            test_accuracy_summary = tf.summary.scalar("test_accuracy"
                                                      , accuracy)

            train_samples_num = len([_ for _ in self.train_data])
            for iteration in range(iterations_num + 1):
                self.session.run(self.optimizer
                            , feed_dict={self.x: self.train_data.reshape(train_samples_num
                                                                               , self.hidden_layers_num
                                                                               , self.features_num)
                                         , self.size_of_x: train_samples_num
                                         , self.y: self.train_labels.reshape(train_samples_num
                                                                             , 1)})
                if iteration % 10 == 0:
                    print(iteration)
                    summary = self.session.run(merged_summary
                                          , feed_dict={self.x: self.train_data.reshape(train_samples_num
                                                                                             , self.hidden_layers_num
                                                                                             , self.features_num)
                                                       , self.size_of_x: train_samples_num
                                                       , self.y: self.train_labels.reshape(train_samples_num
                                                                                                 , 1)})
                    writer.add_summary(summary, iteration)
                    summary = self.session.run(train_accuracy_summary
                                          , feed_dict={self.x: self.train_data.reshape(train_samples_num
                                                                                             , self.hidden_layers_num
                                                                                             , self.features_num)
                                                       , self.size_of_x: train_samples_num
                                                       , self.y: self.train_labels.reshape(train_samples_num
                                                                                                 , 1)})
                    writer.add_summary(summary
                                       , iteration)
                    summary = self.session.run(test_accuracy_summary
                                          , feed_dict={self.x: self.test_data.reshape(test_samples_num
                                                                                            , self.hidden_layers_num
                                                                                            , self.features_num)
                                                       , self.size_of_x: test_samples_num
                                                       , self.y: self.test_labels.reshape(test_samples_num
                                                                                                 , 1)})
                    writer.add_summary(summary
                                       , iteration)


def sign(x):
    return int(x > 0) - int(x < 0)


def check_accuracy(classifiers, samples, labels):
    correct_count = 0.0

    for idx, sample in enumerate(samples):
        label = labels[idx]
        predicted_label = 0
        max_score = -CLASSES_NUM

        for i, ith_classifiers in enumerate(classifiers):
            curr_score = 0

            for classifier in ith_classifiers:
                curr_score += sign(eval_sample(classifier, sample))
            if curr_score > max_score:
                max_score = curr_score
                predicted_label = i + 1
        if predicted_label == label:
            correct_count += 1
    return correct_count / len(samples)


def eval_sample(classifier, sample):
    return classifier.prediction.eval(session=classifier.session
                                      , feed_dict={classifier.x: sample.reshape(1
                                                                                , classifier.hidden_layers_num
                                                                                , classifier.features_num)
                                                   , classifier.size_of_x: 1})


def main():
    db = SVM_DataBase(80)
    train_samples = db.train["data"]
    train_labels = db.train["labels"]
    test_samples = db.test["data"]
    test_labels = db.test["labels"]

    classifier_list = []
    for i in range(CLASSES_NUM):
        classifier_list.append([])
        for j in range(CLASSES_NUM):
            if i != j:
                db.train["data"] = np.copy(train_samples)
                db.train["labels"] = np.copy(train_labels)
                db.test["data"] = np.copy(test_samples)
                db.test["labels"] = np.copy(test_labels)
                db.filter_labels(i + 1, j + 1)
                classifier = SVMModel(db
                                      , 256
                                      , 0.001
                                      , 0.1
                                      , 0.5)
                classifier_list[i].append(classifier)
                classifier.run(1000)

    print(check_accuracy(classifier_list
                         , test_samples
                         , test_labels))


main()
