# Image processing
INPUT_COMMON_DIR = "C:\\Users\\alandree\\Desktop\\emotions\\kasrl\\ck+"
INPUT_COMMON_DIR_IMGS = "C:\\Users\\alandree\\Desktop\\emotions\\kasrl\\ck+\\cohn-kanade-images"
INPUT_COMMON_DIR_EMOTIONS = "C:\\Users\\alandree\\Desktop\\emotions\\kasrl\\ck+\\Emotion"
INPUT_COMMON_DIR_FACS = "C:\\Users\\alandree\\Desktop\\emotions\\kasrl\\ck+\\FACS"
INPUT_COMMON_DIR_LANDMARKS = "C:\\Users\\alandree\\Desktop\\emotions\\kasrl\\ck+\\Landmarks"
INPUT_SINGLE_IMG_FPATH_TEST = "C:\\Users\\alandree\\Desktop\\emotions\\kasrl\\ck+\\cohn-kanade-images\\S005\\001\\S005_001_00000001.png"
DEFAULT_IMG_SIZE = (640, 480)
TEST_CNT = 30
ALL_DATA_JSON = "all_data.json"
ERROR_FNAME = "logs/errors.log"
NORMALIZED_DIR = "normalized_all/"
INFO_LOG = "logs/info.log"

# mode_rgb = 'C:\\Users\\alandree\\Desktop\\emotions\\kasrl\\ck+\\cohn-kanade-images\\S053\\003\\S053_003_00000001.png'
# mode_l = 'C:\\Users\\alandree\\Desktop\\emotions\\kasrl\\ck+\\cohn-kanade-images\\S005\\001\\S005_001_00000001.png'
# size_640_490 = 'C:\\Users\\alandree\\Desktop\\emotions\\kasrl\\ck+\\cohn-kanade-images\\S005\\001\\S005_001_00000001.png'
# size_640_480 = 'C:\\Users\\alandree\\Desktop\\emotions\\kasrl\\ck+\\cohn-kanade-images\\S053\\003\\S053_003_00000001.png'
# size_720_480 = 'C:\\Users\\alandree\\Desktop\\emotions\\kasrl\\ck+\\cohn-kanade-images\\S501\\001\\S501_001_00000001.png'
# facs_fname = 'C:\\Users\\alandree\\Desktop\\emotions\\kasrl\\ck+\\FACS\\S011\\004\\S011_004_00000021_facs.txt'

# Video
LANDMARK_FNAME = "landmark_frame_%s.txt"

TRAINED_MODELS = 'trained_models/'
DEMO_VIDEO_FNAME = 'videos/WIN_20180723_02_52_55_Pro.mp4'

def write_log(log_fname, s):
    f = open(log_fname, "a+")
    f.write(s + "\n")
    f.close()
