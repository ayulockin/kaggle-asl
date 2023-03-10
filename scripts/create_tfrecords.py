import os
import json
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
print(tf.__version__)
import tensorflow_io as tfio
print(tfio.__version__)


# train.csv file
df = pd.read_csv("train.csv")

# Get labels 2 id
with open("sign_to_prediction_index_map.json") as f:
    label2id = json.load(f)
    
df["sign"] = df["sign"].apply(lambda sign: label2id[sign])

paths = df.path.values
labels = df.sign.values

tfrecords_dir = "data/tfrecords"

num_samples = 4096
num_tfrecords = len(df) // num_samples
if len(df) % num_samples:
    num_tfrecords += 1  # add one record if there are any remaining samples

if not os.path.exists(tfrecords_dir):
    os.makedirs(tfrecords_dir)  # creating TFRecords output folder


skf = StratifiedKFold(n_splits=num_tfrecords, random_state=None, shuffle=False)

stratified_labels = {}

for i, (_, test_index) in enumerate(skf.split(paths, labels)):
    # print(f"Fold {i}:")
    # print(f"  Test:  index={test_index}")

    stratified_labels[i] = test_index


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
        out_type=tf.float32,
    )


def create_example(n_frames, sequence, label):
    feature = {
        "n_frames": float_feature(n_frames),
        "frames": bytes_feature(serialize_sequence(frames)),
        "label": int64_feature(label),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    feature_description = {
        "n_frames": tf.io.FixedLenFeature([], tf.float32),
        "frames": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    
    n_frames = example["n_frames"]
    label = tf.one_hot(example["label"], depth=250)
    frames = tf.reshape(parse_sequence(example["frames"]), shape=(n_frames, 543, 3))
    
    return example


for k, v in tqdm(stratified_labels.items(), desc="Creating TFRecords: "):
    split_paths = paths[v]
    split_labels = labels[v]

    with tf.io.TFRecordWriter(
        tfrecords_dir + f"/stratified_split_{k}.tfrec"
    ) as writer:
        for path, label in tqdm(zip(split_paths, split_labels), desc="Writing TFRecords: "):
            # without filling nan values
            frames = pd.read_parquet(path)[["x", "y", "z"]].values.astype(np.float32)
            n_frames = len(frames)/543

            example = create_example(
                n_frames,
                frames,
                label
            )

            writer.write(example.SerializeToString())
