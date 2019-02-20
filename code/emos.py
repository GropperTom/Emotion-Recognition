import numpy as np
import matplotlib.cm as cm

EMO_STR = np.array(["Anger", "Contempt", "Disgust", "Fear", "Happy", "Sadness", "Surprise"])
EMO_COLORS = cm.rainbow(np.linspace(0, 1, len(EMO_STR)))
EMO_NUM = len(EMO_STR)

emo_to_index = lambda emo: int(emo-1)
index_to_emo = lambda index: int(index + 1)

def emo_file_to_vector(full_emo_fname):
    # File:
    # 3.0000000e+00
    # Vector: [0, 0, 1, 0, 0, 0, 0] (7 emotions)
    lst_emos = map(lambda x: [float(y) for y in x.split()], open(full_emo_fname).readlines())
    v = np.zeros(EMO_NUM)
    for emos_line in lst_emos:
        for emo in emos_line:
            v[emo_to_index(emo)] = 1
    return v

def vector_emo(v):
    # Vector: [0, 0, 1, 0, 0, 0, 0]
    # Emotions: [3] = ["Disgust"]
    indices = np.nonzero(v)
    return indices + np.ones(len(indices)), EMO_STR[indices]
