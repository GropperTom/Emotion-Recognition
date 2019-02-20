import tensorflow as tf
import numpy as np
from DataBase import RNN_DataBase

TENSORBOARD_DIR = "tensorboard\RNN"
#tensorboard --logdir=C:\Users\gropp\PycharmProjects\ML_Project\tensorboard\RNN


class RNNModel:
    def __init__(self, db, neurons_num, learning_rate, dropout_percent=None):
        self.db = db
        self.graph = tf.Graph()
        self.hidden_layers_num = self.db.get_hidden_layers_num()
        self.features_num = db.get_features_num()
        self.labels_num = db.get_labels_num()
        with self.graph.as_default():
            self.session = None
            with tf.name_scope("RNN_model"):
                self.size_of_x = tf.placeholder(tf.int32, shape=())
                self.x = tf.placeholder(tf.float32
                                        , [None, db.max_length, self.features_num]
                                        , name="data")
                self.y = tf.placeholder(tf.float32
                                        , [None, self.labels_num]
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
                weights = tf.Variable(tf.truncated_normal([neurons_num, self.labels_num]
                                                          , stddev=0.1)
                                      , name="weights")
                biases = tf.Variable((tf.ones([self.labels_num]) * 0.1)
                                     , name="biases")
                logits = tf.matmul(last
                                   , weights) + biases
                self.prediction = tf.nn.softmax(logits
                                                , name="prediction")
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits
                                                                                 , labels=self.y)
                                      , name="loss")
                with tf.name_scope("train"):
                    self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss
                                                                                    , name="optimizer")

                tf.summary.histogram("weights_histogram"
                                     , weights)
                tf.summary.histogram("biases_histogram"
                                     , biases)
                tf.summary.scalar("loss_summary"
                                  , loss)

    def gather_relevant(self, output, lengths):
        index = tf.range(self.size_of_x) * self.db.max_length + (lengths - 1)
        relevant = tf.gather(tf.reshape(output, [-1, output.get_shape()[2]]), index)
        return relevant

    def run(self, iterations_num):
        with self.graph.as_default():
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(TENSORBOARD_DIR)
            writer.add_graph(self.graph)

            merged_summary = tf.summary.merge_all()

            test_samples_num = self.db.get_test_samples_num()
            correct_prediction = tf.equal(tf.argmax(self.y
                                                    , 1)
                                          , tf.argmax(self.prediction
                                                      , 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction
                                              , tf.float32)) * 100
            train_accuracy_summary = tf.summary.scalar("train_accuracy"
                                                       , accuracy)
            test_accuracy_summary = tf.summary.scalar("test_accuracy"
                                                      , accuracy)

            train_samples_num = self.db.get_train_samples_num()
            for iteration in range(iterations_num + 1):
                self.session.run(self.optimizer
                            , feed_dict={self.x: self.db.train["data"].reshape(train_samples_num
                                                                               , self.hidden_layers_num
                                                                               , self.features_num)
                                         , self.size_of_x: train_samples_num
                                         , self.y: self.db.train["labels"].reshape(train_samples_num
                                                                                   , self.labels_num)})
                if iteration % 10 == 0:
                    print(iteration)
                    summary = self.session.run(merged_summary
                                          , feed_dict={self.x: self.db.train["data"].reshape(train_samples_num
                                                                                             , self.hidden_layers_num
                                                                                             , self.features_num)
                                                       , self.size_of_x: train_samples_num
                                                       , self.y: self.db.train["labels"].reshape(train_samples_num
                                                                                                 , self.labels_num)})
                    writer.add_summary(summary, iteration)
                    summary = self.session.run(train_accuracy_summary
                                          , feed_dict={self.x: self.db.train["data"].reshape(train_samples_num
                                                                                             , self.hidden_layers_num
                                                                                             , self.features_num)
                                                       , self.size_of_x: train_samples_num
                                                       , self.y: self.db.train["labels"].reshape(train_samples_num
                                                                                                 , self.labels_num)})
                    writer.add_summary(summary, iteration)
                    summary = self.session.run(test_accuracy_summary
                                          , feed_dict={self.x: self.db.test["data"].reshape(test_samples_num
                                                                                            , self.hidden_layers_num
                                                                                            , self.features_num)
                                                       , self.size_of_x: test_samples_num
                                                       , self.y: self.db.test["labels"].reshape(test_samples_num
                                                                                                , self.labels_num)})
                    writer.add_summary(summary, iteration)


def main():
    db = RNN_DataBase(80)
    model = RNNModel(db, 256, 0.001, dropout_percent=0.5)
    model.run(3000)


main()
