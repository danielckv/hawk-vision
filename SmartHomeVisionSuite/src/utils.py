import datetime
import os

import cv2

CURRENT_DIR_APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CURRENT_SNAPSHOTS_DIR = os.path.join(CURRENT_DIR_APP, ".snapshots")
LAST_SAVED_FRAME_TIME = 0
CONFIG = {}


def save_frame_to_jpeg(frame):
    global LAST_SAVED_FRAME_TIME
    if not os.path.exists(CURRENT_SNAPSHOTS_DIR):
        os.makedirs(CURRENT_SNAPSHOTS_DIR)

    if not should_save_frame_period():
        return

    if frame is None:
        return

    filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"
    filename = os.path.join(CURRENT_SNAPSHOTS_DIR, filename)
    cv2.imwrite(filename, frame)
    LAST_SAVED_FRAME_TIME = datetime.datetime.now()


def should_save_frame_period():
    global LAST_SAVED_FRAME_TIME
    current_time = datetime.datetime.now()
    if LAST_SAVED_FRAME_TIME == 0:
        return True

    time_diff = current_time - LAST_SAVED_FRAME_TIME
    return time_diff.seconds >= CONFIG['saveFramePeriod']


def load_config_yaml():
    global CONFIG
    import yaml
    with open(os.path.join(CURRENT_DIR_APP, 'config.yaml'), 'r') as stream:
        try:
            CONFIG = yaml.safe_load(stream)
            return CONFIG
        except yaml.YAMLError as exc:
            print(exc)
