import os
import numpy as np
import math
import random

DB_FOLDER = "ck+"


def normalize(lst, max_dist):
    for element in lst:
        element[0] /= max_dist
    return lst


def get_vector(p1, p2):
    delta_x = math.fabs(p1[0] - p2[0])
    delta_y = math.fabs(p1[1] - p2[1])
    return [math.sqrt(delta_x*delta_x + delta_y*delta_y), math.degrees(math.atan2(delta_x, delta_y))]


class RNN_DataBase(object):
    def __init__(self, train_percent, db_folder=DB_FOLDER):
        self.train_percent = train_percent
        self.max_length = 0
        self.train = {"data": [], "labels": []}
        self.test = {"data": [], "labels": []}

        for subject in os.listdir(db_folder + "//Landmarks//"):
            for session in os.listdir(db_folder + "//Landmarks//" + subject + "//"):
                new_session = []

                labels_path = db_folder + "//Emotion//" + subject + "//" + session + "//"
                try:
                    label_file = open(labels_path + os.listdir(labels_path)[0])

                    for landmark in os.listdir(db_folder + "//Landmarks//" + subject + "//" + session + "//"):
                        landmark_file = open(db_folder + "//Landmarks//" + subject + "//" + session + "//" + landmark
                                             , 'r')
                        landmark_center = np.array([0, 0])
                        temp = []
                        for line in landmark_file:
                            x_coord = float(line[3:16])
                            y_coord = float(line[19:32])
                            temp.append([x_coord, y_coord])
                            landmark_center[0] += x_coord
                            landmark_center[1] += y_coord

                        landmark_center = landmark_center / 68
                        max_dist = 0
                        for i, point in enumerate(temp):
                            vector = get_vector(landmark_center, point)
                            temp[i] = vector
                            if vector[0] > max_dist:
                                max_dist = vector[0]

                        new_landmark = []
                        normalize(temp, max_dist)
                        for point in temp:
                            new_landmark.append(point[0])
                            new_landmark.append(point[1])

                        new_session.append(new_landmark)
                        landmark_file.close()

                    length = len(new_session)
                    if length > self.max_length:
                        self.max_length = length

                    label = int(label_file.readline()[3:4])
                    one_hot_encode = [1 if i == label - 1 else 0 for i in range(7)]
                    if random.randint(0, 101) <= train_percent:
                        self.train["data"].append(new_session)
                        self.train["labels"].append(one_hot_encode)

                    else:
                        self.test["data"].append(new_session)
                        self.test["labels"].append(one_hot_encode)
                    label_file.close()
                except (FileNotFoundError, IndexError):
                    pass

        for i, train_sample in enumerate(self.train["data"]):
            train_sample += [[0 for _ in range(136)] for _ in range(self.max_length - len(train_sample))]

        for i, test_sample in enumerate(self.test["data"]):
            test_sample += [[0 for _ in range(136)] for _ in range(self.max_length - len(test_sample))]

        self.train["data"] = np.array(self.train["data"])
        self.train["labels"] = np.array(self.train["labels"])
        self.test["data"] = np.array(self.test["data"])
        self.test["labels"] = np.array(self.test["labels"])

    def get_features_num(self):
        return len(self.train["data"][0][0])

    def get_labels_num(self):
        return len(self.train["labels"][0])

    def get_hidden_layers_num(self):
        return len(self.train["data"][0])

    def get_train_samples_num(self):
        return len([_ for _ in self.train["data"]])

    def get_test_samples_num(self):
        return len([_ for _ in self.test["data"]])


class SVM_DataBase(object):
    def __init__(self, train_percent, db_folder=DB_FOLDER):
        self.train_percent = train_percent
        self.max_length = 0
        self.train = {"data": [], "labels": []}
        self.test = {"data": [], "labels": []}

        for subject in os.listdir(db_folder + "//Landmarks//"):
            for session in os.listdir(db_folder + "//Landmarks//" + subject + "//"):
                new_session = []

                labels_path = db_folder + "//Emotion//" + subject + "//" + session + "//"
                try:
                    label_file = open(labels_path + os.listdir(labels_path)[0])

                    for landmark in os.listdir(db_folder + "//Landmarks//" + subject + "//" + session + "//"):
                        landmark_file = open(db_folder + "//Landmarks//" + subject + "//" + session + "//" + landmark
                                             , 'r')
                        landmark_center = np.array([0, 0])
                        temp = []
                        for line in landmark_file:
                            x_coord = float(line[3:16])
                            y_coord = float(line[19:32])
                            temp.append([x_coord, y_coord])
                            landmark_center[0] += x_coord
                            landmark_center[1] += y_coord

                        landmark_center = landmark_center / 68
                        max_dist = 0
                        for i, point in enumerate(temp):
                            vector = get_vector(landmark_center, point)
                            temp[i] = vector
                            if vector[0] > max_dist:
                                max_dist = vector[0]

                        new_landmark = []
                        normalize(temp, max_dist)
                        for point in temp:
                            new_landmark.append(point[0])
                            new_landmark.append(point[1])

                        new_session.append(new_landmark)
                        landmark_file.close()

                    length = len(new_session)
                    if length > self.max_length:
                        self.max_length = length

                    label = int(label_file.readline()[3:4])
                    if random.randint(0, 101) <= train_percent:
                        self.train["data"].append(new_session)
                        self.train["labels"].append(label)

                    else:
                        self.test["data"].append(new_session)
                        self.test["labels"].append(label)
                    label_file.close()
                except (FileNotFoundError, IndexError):
                    pass

        for train_sample in self.train["data"]:
            train_sample += [[0 for _ in range(136)] for _ in range(self.max_length - len(train_sample))]

        for test_sample in self.test["data"]:
            test_sample += [[0 for _ in range(136)] for _ in range(self.max_length - len(test_sample))]

        self.train["data"] = np.array(self.train["data"])
        self.train["labels"] = np.array(self.train["labels"])
        self.test["data"] = np.array(self.test["data"])
        self.test["labels"] = np.array(self.test["labels"])

    def get_features_num(self):
        return len(self.train["data"][0][0])

    def get_hidden_layers_num(self):
        return len(self.train["data"][0])

    def get_train_samples_num(self):
        return len([_ for _ in self.train["data"]])

    def get_test_samples_num(self):
        return len([_ for _ in self.test["data"]])

    def filter_labels(self, label1, label2):
        delete_arr = []
        for i, label in enumerate(self.train["labels"]):
            if label == label1:
                self.train["labels"][i] = 1
            elif label == label2:
                self.train["labels"][i] = -1
            else:
                delete_arr.append(i)
        self.train["data"] = np.delete(self.train["data"], delete_arr, axis=0)
        self.train["labels"] = np.delete(self.train["labels"], delete_arr, axis=0)

        delete_arr = []
        for i, label in enumerate(self.test["labels"]):
            if label == label1:
                self.test["labels"][i] = 1
            elif label == label2:
                self.test["labels"][i] = -1
            else:
                delete_arr.append(i)
        self.test["data"] = np.delete(self.test["data"], delete_arr, axis=0)
        self.test["labels"] = np.delete(self.test["labels"], delete_arr, axis=0)


#test = DataBase(100)
#print(test.get_hidden_layers_num())