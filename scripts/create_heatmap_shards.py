import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import re
import cv2
import math
import time
import json
import string
import random
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import multiprocessing 
from argparse import Namespace

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

tf.config.set_visible_devices([], 'GPU')

# train.csv file
df = pd.read_csv("data/train.csv")

# Get labels 2 id
with open("data/sign_to_prediction_index_map.json") as f:
    label2id = json.load(f)


def add_path(row):
    return "data/"+row.path
    
df["sign"] = df["sign"].apply(lambda sign: label2id[sign])
df["path"] = df.apply(lambda row: add_path(row), axis=1)

AUTOTUNE = tf.data.AUTOTUNE

NUM_JOINTS = 81
IMG_H = 48
IMG_W = 48
tfrecords_dir = "../data/tfrecord_heatmaps_tmp"
os.makedirs(tfrecords_dir, exist_ok=True)


LIP = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
]

RIGHT_EYE = [
    246, 161, 160, 159, 158, 157, 173,
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    247, 30, 29, 27, 28, 56, 190,
    130, 25, 110, 24, 23, 22, 26, 112, 243,
    113, 225, 224, 223, 222, 221, 189,
    226, 31, 228, 229, 230, 231, 232, 233, 244,
    143, 111, 117, 118, 119, 120, 121, 128, 245,
]

LEFT_EYE = [
    466, 387, 386, 385, 384, 398,
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    467, 260, 259, 257, 258, 286, 414,
    359, 255, 339, 254, 253, 252, 256, 341, 463,
    342, 445, 444, 443, 442, 441, 413,
    446, 261, 448, 449, 450, 451, 452, 453, 464,
    372, 340, 346, 347, 348, 349, 350, 357, 465,
]

LEFT_HAND = [
    468, 469, 470, 471, 472, 473, 474, 475,
    476, 477, 478, 479, 480, 481, 482, 483,
    484, 485, 486, 487, 488
]

RIGHT_HAND = [
    522, 523, 524, 525, 526, 527, 528, 529,
    530, 531, 532, 533, 534, 535, 536, 537,
    538, 539, 540, 541, 542
]

POSE = [
    489, 490, 491, 492, 493, 494, 495, 496, 497,
    498, 499, 500, 501, 502, 503, 504, 505, 512,
    513, 514, 515, 516, 517, 518, 519, 520, 521
]


def generate_a_heatmap(arr, centers, max_values):
    """Generate pseudo heatmap for one keypoint in one frame.

    Args:
        arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
        centers (np.ndarray): The coordinates of corresponding keypoints (of multiple persons). Shape: M * 2.
        max_values (np.ndarray): The max values of each keypoint. Shape: M.

    Returns:
        np.ndarray: The generated pseudo heatmap.
    """

    sigma = 0.9
    img_h, img_w = arr.shape

    for center, max_value in zip(centers, max_values):
        mu_x, mu_y = center[0], center[1]
        if not (np.isnan(mu_x) and np.isnan(mu_y)):
            # scale
            mu_x = min(math.floor(mu_x * img_w), img_w - 1)
            mu_y = min(math.floor(mu_y * img_h), img_h - 1)

            st_x = max(int(mu_x - 2 * sigma), 0)
            ed_x = min(int(mu_x + 2 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 2 * sigma), 0)
            ed_y = min(int(mu_y + 2 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
            patch = patch * max_value
            arr[st_y:ed_y, st_x:ed_x] = np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)


def get_3d_heatmap(ret, human_kps, num_frames):
    
    for i, frame in enumerate(range(num_frames)):
        arr = ret[i]
        human = human_kps[i]

        x, y = human[:,:1], human[:,1:2]

        # TODO: Normalize the whole sequence together
        x = (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
        y = (y-np.nanmin(y))/(np.nanmax(y)-np.nanmin(y))

        human = np.squeeze(np.array(list(zip(x, y))), axis=-1)

        kps = np.expand_dims(human, axis=0)
        all_kpscores = np.ones((1,num_frames,NUM_JOINTS), dtype=np.float32)
        kpscores = np.ones_like(all_kpscores[:, 0])

        num_kp = kps.shape[1]
        for i in range(num_kp):
            generate_a_heatmap(arr[i], kps[:, i], kpscores[:, i])
            
    return ret


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))


def serialize_sequence(sequence):
    """Serialize the multidimentional tensor"""
    return tf.io.serialize_tensor(sequence)


def parse_sequence(serialized_sequence):
    return tf.io.parse_tensor(
        serialized_sequence,
        out_type=tf.float16,
    )


def create_example(n_frames, sequence, label):
    feature = {
        "n_frames": float_feature(n_frames),
        "frames": bytes_feature(serialize_sequence(sequence)),
        "label": int64_feature(label),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

