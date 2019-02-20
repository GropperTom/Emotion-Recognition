
import cv2
import imutils
import time
import dlib
import numpy as np

from landmarks import land2coords
from softmax import *
from landmarks import normalize_landmark_sasha

import matplotlib.pyplot as plt
from drawnow import drawnow
from emos import *

# sasha 2do: init_softmax_pretrained return W,b should return dict{'W':..., 'b': ...}
# prediction_cb should get params in this way(dict rather than separate params)

MODELS_PROCESS_DATA = {"softmax": {"init_function": init_softmax_pretrained, "prediction_cb": predict_softmax_}}

def main_run_video_softmax(video_fname = DEMO_VIDEO_FNAME):
    '''
    :param video_fname: relative path of to video file, for example: 'videos/WIN_20180723_02_52_55_Pro.mp4'
    :return:            draws SOFTMAX results: 1) landmarked vide - frame after frame 2) animated graph
    :2do:               delete this function
    '''
    # Init the plot
    def make_fig():
        for i in range(EMO_NUM):
            plt.scatter(x, [all_emos[i] for all_emos in y], color=EMO_COLORS[i], label=EMO_COLORS[i])

    x = list()
    y = list()

    plt.ion()  # enable interactivity
    fig, ax = plt.subplots()

    # loading dlib's Hog Based face detector
    face_detector = dlib.get_frontal_face_detector()

    # loading dlib's 68 points-shape-predictor
    # get file:shape_predictor_68_face_landmarks.dat from
    # link: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # 0 means your default web cam
    vid = cv2.VideoCapture(video_fname)

    # Init softmax
    W, b = init_softmax_pretrained()

    frames_cnt = 1
    all_landmarks = []
    all_predictions = []
    time_start = time.time()

    while True:
        try:
            _, frame = vid.read()

            # resizing frame
            # you can use cv2.resize but I recommend imutils because its easy to use
            frame = imutils.resize(frame, width=400)

            # grayscale conversion of image because it is computationally efficient
            # to perform operations on single channeled (grayscale) image
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detecting faces
            face_boundaries = face_detector(frame_gray, 0)

            for (enum, face) in enumerate(face_boundaries):
                # let's first draw a rectangle on the face portion of image
                x_face = face.left()
                y_face = face.top()
                w = face.right() - x_face
                h = face.bottom() - y_face
                # Drawing Rectangle on face part
                cv2.rectangle(frame, (x_face, y_face), (x_face + w, y_face + h), (120, 160, 230), 2)

                # Now when we have our ROI(face area) let's
                # predict and draw landmarks
                landmarks = landmark_predictor(frame_gray, face)
                # converting co-ordinates to NumPy array
                landmarks = land2coords(landmarks)

                # Save landmarks
                all_landmarks.append(landmarks)

                # Draw landmarks
                for (a, b) in landmarks:
                    # Drawing points on face
                    cv2.circle(frame, (a, b), 2, (255, 0, 0), -1)

                # Writing face number on image
                cv2.putText(frame, "Face :{}".format(enum + 1), (x_face - 10, y_face - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 128), 2)

                # Predict
                current_prediction_vector = predict_softmax(W, b, normalize_landmark_sasha(np.array(landmarks, dtype = np.float64)))
                print(current_prediction_vector)
                all_predictions.append(current_prediction_vector)

                # Draw the plot
                x.append(frames_cnt)
                y.append(current_prediction_vector)
                drawnow(make_fig)
                # plt.pause(0.05)

                frames_cnt += 1

        except:
            break

        time_end = time.time()
        print("Frame: %s, times_elapsed: %.2f (secs)" % (frames_cnt, time_end-time_start))
        cv2.imshow("frame", frame)

        #  Stop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break;
    plt.show()

    # for (i, single_frame_lmark) in enumerate(all_landmarks):
    #     json.dump(single_frame_lmark.tolist(), codecs.open(LANDMARK_FNAME % i, 'w', encoding='utf-8'),
    #               separators=(',', ':'), sort_keys=True, indent=4)  ### this saves the array in .json format


    # To read the file:
    # landmarks_str = codecs.open(LANDMARK_FNAME % i, 'r', encoding='utf-8').read()
    # landmarks_list = json.loads(landmarks_str)
    # landmarks_arr = np.array(landmarks_list)

def init_pretrained(model):
    if model in MODELS_PROCESS_DATA.keys():
        return {"params": MODELS_PROCESS_DATA[model]["init_function"](),
                "prediction_cb": MODELS_PROCESS_DATA[model]["prediction_cb"]}

def main_run_video_model(model="softmax", video_fname=DEMO_VIDEO_FNAME):
    '''
    :param video_fname: relative path of to video file, for example: 'videos/WIN_20180723_02_52_55_Pro.mp4'
    :param model:       "softmax",
    :return:            draws: 1) landmarked vide - frame after frame 2) animated graph
    :2do:               1) make all simultaniousely2) set labels at the graps (for each color what does it mean)
    '''

    # Init the plot
    def make_fig():
        for i in range(EMO_NUM):
            plt.scatter(x, [all_emos[i] for all_emos in y], color=EMO_COLORS[i], label=EMO_COLORS[i])

    x = list()
    y = list()

    plt.ion()  # enable interactivity
    fig, ax = plt.subplots()

    # loading dlib's Hog Based face detector
    face_detector = dlib.get_frontal_face_detector()

    # loading dlib's 68 points-shape-predictor
    # get file:shape_predictor_68_face_landmarks.dat from
    # link: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # 0 means your default web cam
    vid = cv2.VideoCapture(video_fname)

    # Init softmax
    model_data = init_pretrained(model)

    frames_cnt = 1
    all_landmarks = []
    all_predictions = []
    time_start = time.time()

    while True:
        try:
            _, frame = vid.read()

            # resizing frame
            # you can use cv2.resize but I recommend imutils because its easy to use
            frame = imutils.resize(frame, width=400)

            # grayscale conversion of image because it is computationally efficient
            # to perform operations on single channeled (grayscale) image
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detecting faces
            face_boundaries = face_detector(frame_gray, 0)

            for (enum, face) in enumerate(face_boundaries):
                # let's first draw a rectangle on the face portion of image
                x_face = face.left()
                y_face = face.top()
                w = face.right() - x_face
                h = face.bottom() - y_face
                # Drawing Rectangle on face part
                cv2.rectangle(frame, (x_face, y_face), (x_face + w, y_face + h), (120, 160, 230), 2)

                # Now when we have our ROI(face area) let's
                # predict and draw landmarks
                landmarks = landmark_predictor(frame_gray, face)
                # converting co-ordinates to NumPy array
                landmarks = land2coords(landmarks)

                # Save landmarks
                all_landmarks.append(landmarks)

                # Draw landmarks
                for (a, b) in landmarks:
                    # Drawing points on face
                    cv2.circle(frame, (a, b), 2, (255, 0, 0), -1)

                # Writing face number on image
                cv2.putText(frame, "Face :{}".format(enum + 1), (x_face - 10, y_face - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 128), 2)

                # Predict
                #current_prediction_vector = alg_func(W, b, normalize_landmark_sasha(np.array(landmarks, dtype = np.float64)))
                current_prediction_vector = model_data["prediction_cb"](model_data["params"], normalize_landmark_sasha(np.array(landmarks, dtype = np.float64)))
                print(current_prediction_vector)
                all_predictions.append(current_prediction_vector)

                # Draw the plot
                x.append(frames_cnt)
                y.append(current_prediction_vector)
                drawnow(make_fig)
                # plt.pause(0.05)

                frames_cnt += 1

        except:
            break

        time_end = time.time()
        print("Frame: %s, times_elapsed: %.2f (secs)" % (frames_cnt, time_end-time_start))
        cv2.imshow("frame", frame)

        #  Stop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break;
    plt.show()

def get_plot_data_model(video_fname, model):
    '''
    :param video_fname: relative path of to video file, for example: 'videos/WIN_20180723_02_52_55_Pro.mp4'
    :param model:       "softmax",
    :return:            vector of predictions for each video frame (each preditction is vector of 7 emotion's intensities)
    '''

    # Init
    all_predictions_vectors = []
    model_data = init_pretrained(model)

    # Make predictions for each landmark:
    for img_normalized_lmark in get_normalized_landmarks_from_video(video_fname):
        # Predict for this frame
        current_prediction_vector = model_data["prediction_cb"](model_data["params"], img_normalized_lmark)
        all_predictions_vectors.append(current_prediction_vector)

    return all_predictions_vectors

def tune_all():
    from ada_boost import ab_tuning
    from softmax import tune_softmax
    ab_tuning()
    tune_softmax()
