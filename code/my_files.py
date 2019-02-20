from datetime import datetime
import json
from common_defs import *
import os

def stats_files_per_folder(main_dir = INPUT_COMMON_DIR_IMGS):
    files_num_lst = []
    for root, directories, filenames in os.walk(main_dir):
        for directory in directories:
            full_path = os.path.join(root, directory)
            files_num = len([x for x in os.listdir(full_path) if x[-3:] == 'png'])
            if files_num > 0:
                print("%s - %s" % (full_path, files_num))
                files_num_lst.append(len([x for x in os.listdir(full_path) if x[-3:] == 'png']))
    if len(files_num_lst) > 0:
        print("Min files: %d, max files: %d" % (min(files_num_lst), max(files_num_lst)))
    else:
        print("All subfolders are empty")

def get_emo_for_img_dir(full_path_dir):
    from emos import emo_to_index, EMO_STR
    emo_path = full_path_dir.replace("cohn-kanade-images", "Emotion")
    emo_files = os.listdir(emo_path)
    if len(emo_files) == 0:
        print("empty emo_path: %s" % emo_path )
        return "None"
    else:
        f = open(os.path.join(emo_path, emo_files[0]), "r")
        emo_str = f.read().strip()
        emo = int(float(emo_str))
    return EMO_STR[emo_to_index(emo)]

def stats_img_files_per_emo(main_dir = INPUT_COMMON_DIR_IMGS):
    stats = {}
    for root, directories, filenames in os.walk(main_dir):
        for directory in directories:
            sdir = os.path.join(root, directory) # sdir = 'C:\\Users\\alandree\\Desktop\\emotions\\kasrl\\ck+\\cohn-kanade-images\\S014'
            for subdir in os.listdir(sdir):
                try:# '001', '002', ...
                    full_path = os.path.join(sdir, subdir)
                    files_num = len([x for x in os.listdir(full_path) if x[-3:] == 'png'])
                    if files_num > 0:
                        emo = get_emo_for_img_dir(full_path)
                        if emo in stats.keys():
                            stats[emo] += files_num
                        else:
                            stats[emo] = files_num
                except:
                    continue # meta-dirs like: 'C:\\Users\\alandree\\Desktop\\emotions\\kasrl\\ck+\\cohn-kanade-images\\S005\\.DS_Store'
    return stats

def stats_sequences_per_emo(main_dir = INPUT_COMMON_DIR_IMGS):
    stats = {}
    for root, directories, filenames in os.walk(main_dir):
        for directory in directories:
            sdir = os.path.join(root, directory) # sdir = 'C:\\Users\\alandree\\Desktop\\emotions\\kasrl\\ck+\\cohn-kanade-images\\S014'
            for subdir in os.listdir(sdir):
                try:# '001', '002', ...
                    full_path = os.path.join(sdir, subdir)
                    files_num = len([x for x in os.listdir(full_path) if x[-3:] == 'png'])
                    if files_num > 0:
                        emo = get_emo_for_img_dir(full_path)
                        if emo in stats.keys():
                            stats[emo] += 1
                        else:
                            stats[emo] = 1
                except:
                    continue # meta-dirs like: 'C:\\Users\\alandree\\Desktop\\emotions\\kasrl\\ck+\\cohn-kanade-images\\S005\\.DS_Store'
    return stats


def process_subfolders(callback):
    for root, directories, filenames in os.walk(INPUT_COMMON_DIR_IMGS):
        for directory in directories:
            full_path = os.path.join(root, directory)
            files_num = len([x for x in os.listdir(full_path) if x[-3:] == 'png'])
            if files_num > 0:
                data = get_data_single_folder(full_path)
                callback(data)

def get_data_single_folder(subfolder): # 33 files in the folder
    facs_dir = subfolder.replace("cohn-kanade-images", "FACS")
    landmarks_dir = subfolder.replace("cohn-kanade-images", "landmarks68")
    emo_dir = subfolder.replace("cohn-kanade-images", "Emotion")

    data = {'img_subdir': subfolder,
            'landmark_subdir': landmarks_dir,
            'fnames': [x[:-4] for x in os.listdir(subfolder)]}
    # im        full fname:      os.join(data['img_subdir'], data['fnames'][i]+ "_landmarks.txt")
    # landmarks full fname:      os.join(data['landmark_subdir'], data['fnames'][0] + ".png")

    emo_files = os.listdir(emo_dir)
    if len(emo_files) > 0:
        data["emo_full_fname"] = os.path.join(emo_dir, emo_files[0])

    facs_files = os.listdir(facs_dir)
    if len(facs_files) > 0:
        data["facs_full_fname"] = os.path.join(facs_dir, facs_files[0])

    return data

def log_error(msg):
    f = open(ERROR_FNAME, "a+")
    f.write("%s: %s\n" % (datetime.now(), msg))
    f.close()

def save_to_json(orig_full_fname, fname):
    data = {fname:{}}
    # Emotion
    try:
        emotions_fname = orig_full_fname.replace("cohn-kanade-images", "Emotion").replace(".png", "_emotion.txt")
        emotion = float(open(emotions_fname).read().strip())
        data[fname]["emotion"] = emotion
    except:
        log_error("No Emotions file: %s" % fname)

    # Facs
    try:
        facs_fname = orig_full_fname.replace("cohn-kanade-images", "FACS").replace(".png", "_facs.txt")
        lst_facs = list(map(lambda x: [float(y) for y in x.split()], open(facs_fname).readlines()))
        data[fname]["facs"] = lst_facs
    except:
        log_error("No FACS file: %s" % fname)

    # Landmarks
    try:
        landmarks_fname = orig_full_fname.replace("cohn-kanade-images", "Landmarks").replace(".png", "_landmarks.txt")
        lst_landmarks =  list(map(lambda x: [float(y) for y in x.split()], open(landmarks_fname).readlines()))
        data[fname]["landmarks"] = lst_landmarks
    except:
        log_error("No Landmarks file: %s" % fname)

    # Write to file
    f = open(ALL_DATA_JSON, "a+")
    f.write(json.dumps(data) + "\n")
    f.close()

    # Read from it:
    # f = open(ALL_DATA_JSON, "r")
    # data = [json.loads(x) for x in f.readlines()]
    # data[0][new_fname]['landmarks'][0][0]

def clean_history():
    # Clean normalized images
    for fname in os.listdir(NORMALIZED_DIR):
        os.remove(os.path.join(NORMALIZED_DIR, fname))
    # Clean the json
    if os.path.exists(ALL_DATA_JSON):
        os.remove(ALL_DATA_JSON)
    # Clean the logs
    if os.path.exists(ERROR_FNAME):
        os.remove(ERROR_FNAME)