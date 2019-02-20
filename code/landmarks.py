# Source: http://pythongeeks.net/face-landmark-detection-using-python-and-dlib/
# neccessary imports
import cv2
import imutils
import numpy as np
import dlib
import time
import json, codecs
import math

from common_defs import *
import os
from shutil import copyfile


# Function for creating landmark coordinate list
def land2coords(landmarks, dtype="int"):
    # initialize the list of tuples
    # (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (a, b)-coordinates
    for i in range(0, 68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    # return the list of (a, b)-coordinates
    return coords


# main Function
#if __name__ == "__main__":
def main_video():
    # loading dlib's Hog Based face detector
    face_detector = dlib.get_frontal_face_detector()

    # loading dlib's 68 points-shape-predictor
    # get file:shape_predictor_68_face_landmarks.dat from
    # link: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # 0 means your default web cam
    vid = cv2.VideoCapture('videos/WIN_20180723_02_52_55_Pro.mp4')

    frames_cnt = 1
    all_landmarks = []
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
                x = face.left()
                y = face.top()
                w = face.right() - x
                h = face.bottom() - y
                # Drawing Rectangle on face part
                cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 160, 230), 2)

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
                cv2.putText(frame, "Face :{}".format(enum + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 128), 2)

                frames_cnt += 1
        except:
            break

        time_end = time.time()
        print("Frame: %s, times_elapsed: %.2f (secs)" % (frames_cnt, time_end-time_start))
        cv2.imshow("frame", frame)

        #  Stop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break;

    for (i, single_frame_lmark) in enumerate(all_landmarks):
        json.dump(single_frame_lmark.tolist(), codecs.open(LANDMARK_FNAME % i, 'w', encoding='utf-8'),
                  separators=(',', ':'), sort_keys=True, indent=4)  ### this saves the array in .json format

    # To read the file:
    # landmarks_str = codecs.open(LANDMARK_FNAME % i, 'r', encoding='utf-8').read()
    # landmarks_list = json.loads(landmarks_str)
    # landmarks_arr = np.array(landmarks_list)

def landmark_file_to_vector(fname):
    landmarks_str = codecs.open(fname, 'r', encoding='utf-8').read()
    landmarks_list = json.loads(landmarks_str)
    return np.array(landmarks_list)

def add_fname(fname, new_extention):
    return fname.replace(".png", new_extention)

# Get facial recognition for the image
def get_landmark_files(full_fname, face_detector, landmark_predictor):
    # from landmarks import *
    # face_detector = dlib.get_frontal_face_detector()
    # landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    frame = cv2.imread(full_fname)
    frame = imutils.resize(frame, width=400)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_boundaries = face_detector(frame_gray, 0)

    for (enum, face) in enumerate(face_boundaries):
        # let's first draw a rectangle on the face portion of image
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        # Drawing Rectangle on face part
        cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 160, 230), 2)

        # Now when we have our ROI(face area) let's
        # predict and draw landmarks
        landmarks = landmark_predictor(frame_gray, face)
        # converting co-ordinates to NumPy array
        landmarks = land2coords(landmarks)

        # Draw landmarks
        for (a, b) in landmarks:
            # Drawing points on face
            cv2.circle(frame, (a, b), 2, (255, 0, 0), -1)

        # Writing face number on image
        cv2.putText(frame, "Face :{}".format(enum + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 128), 2)

    # cv2.imwrite(output_fname, frame)
    return frame, landmarks

def main_ck_imgs(is_test = False, landmarks_dir = "landmarks/", landmarks_ping_dir = "landmarks/"):
    input_dir = INPUT_COMMON_DIR_IMGS

    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    time_start = time.time()
    cnt = 0

    for subdir, dirs, files in os.walk(input_dir):
        for fname in files:
            cnt += 1
            try:
                if is_test and cnt > TEST_CNT:
                    break

                full_fname = os.path.join(subdir, fname)
                copyfile(full_fname, landmarks_dir+add_fname(fname, "_orig.png"))

                frame, landmarks = get_landmark_files(full_fname, face_detector, landmark_predictor)

                # Save the image with landmarks
                cv2.imwrite(landmarks_ping_dir+add_fname(fname, "_landmarks.png"), frame)

                # Save landmarks
                json.dump(landmarks.tolist(), codecs.open(landmarks_dir+add_fname(fname, "_landmarks.txt"), 'w', encoding='utf-8'),
                                  separators=(',', ':'), sort_keys=True, indent=4)
            except:
                continue

    time_end = time.time()
    print("Proceeded: %s. Time elapsed: %.2f secs" % (cnt, time_end - time_start))

def normalize(points, max_dist):
    for point in points:
        point[0] /= max_dist
    return points


def get_lm_vector(p1, p2):
    delta_x = math.fabs(p1[0] - p2[0])
    delta_y = math.fabs(p1[1] - p2[1])
    return [math.sqrt(delta_x*delta_x + delta_y*delta_y), math.degrees(math.atan2(delta_x, delta_y))]

def normalized_landmark_from_fname(full_fname):
    landmark_file = open(full_fname, 'r')
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
        vector = get_lm_vector(landmark_center, point)
        temp[i] = vector
        if vector[0] > max_dist:
            max_dist = vector[0]

    new_landmark = []
    normalize(temp, max_dist)
    for point in temp:
        new_landmark.append(point[0])
        new_landmark.append(point[1])
    landmark_file.close()
    return np.array(new_landmark)

# Entire data set to be centered on (0.5, 0.5) with a range of (0,1) in both axes
# All values of coordinates will be in [0.0, 1.0]
# The distance between points is preserved. (?)
def normalized_landmark_from_fname_sasha(full_fname):
    f = open(full_fname, 'r')
    lm_data = f.read()
    f.close()

    lst_of_strings = lm_data.strip().split("\n") # ['2.4885778e+02   1.9971835e+02', '   2.4945797e+02   2.2707980e+02', ...]
    lst_of_floats = [[float(x) for x in s.strip().split()] for s in lst_of_strings] # [[248.85778, 199.71835], [249.45797, 227.0798], ...]
    return normalize_landmark_sasha(lst_of_floats)

def normalize_landmark_sasha(lmark):
    # Init
    xs = [x[0] for x in lmark]
    ys = [x[1] for x in lmark]
    maxX, minX = max(xs), min(xs)
    maxY, minY = max(ys), min(ys)

    # Center the data on the origin:
    x_center, y_center = (maxX + minX) / (1.0 * 2), (maxY + minY) / (1.0 * 2)
    lmark -= np.array([np.array([x_center, y_center])])

    # Scale it down by the same amount in both dimensions, such that the larger of the two dimensions
    # Need to scale both xs and ys in order to preserve the distance
    scale = max(maxX - minX, maxY - minY)
    lmark /= np.array([scale, scale])

    # Move all points to be around (0.5, 0.5)
    lmark += np.array([0.5, 0.5])

    return list(lmark.flatten())

def get_normalized_landmarks_from_video(video_fname):
    # loading dlib's Hog Based face detector
    face_detector = dlib.get_frontal_face_detector()

    # loading dlib's 68 points-shape-predictor
    # get file:shape_predictor_68_face_landmarks.dat from
    # link: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # 0 means your default web cam
    vid = cv2.VideoCapture(video_fname)

    all_landmarks = []
    frame_cnt = 0
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
                    # Find landmarks for the single image frame
                    landmarks = landmark_predictor(frame_gray, face)

                    # Convert coordinates to NumPy array
                    landmarks = land2coords(landmarks)

                    # Normalize it
                    normalized = normalize_landmark_sasha(np.array(landmarks, dtype=np.float64))

                    # Save landmarks
                    all_landmarks.append(normalized)

                frame_cnt += 1
        except:
            print("Exception is thrown")
            break

    return all_landmarks