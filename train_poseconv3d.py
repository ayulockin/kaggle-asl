import os
import re
import json
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from argparse import Namespace
import matplotlib.pyplot as plt

import tensorflow as tf
print(tf.__version__)
import tensorflow_io as tfio
print(tfio.__version__)
import tensorflow_addons as tfa

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.regularizers import L2

import wandb
from wandb.keras import WandbMetricsLogger
from wandb.keras import WandbModelCheckpoint


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


data_path = "data/tfrecord_heatmaps"


def natural_keys(text):
    ""
    def atoi(text):
        return int(text) if text.isdigit() else text
    
    return [atoi(c) for c in re.split(r'(\d+)', text)]

tfrecords = sorted(glob(f"{data_path}/*.tfrec"), key=natural_keys)


configs = Namespace(
    batch_size = 128,
    epochs = 30,
    learning_rate = 1e-3,
    label_smoothing=0.3, # change 1
    num_steps=0.7,
)

train_tfrecords, valid_tfrecords = tfrecords[:20], tfrecords[20:]
print(len(train_tfrecords), len(valid_tfrecords))


def parse_sequence(serialized_sequence):
    return tf.io.parse_tensor(
        serialized_sequence,
        out_type=tf.float16,
    )


def parse_tfrecord_fn(example):
    feature_description = {
        "n_frames": tf.io.FixedLenFeature([], tf.float32),
        "frames": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    return tf.io.parse_single_example(example, feature_description)


def preprocess_frames(frames):
    """This is where different preprocessing logics will be experimented."""
    frames = (frames - tf.reduce_min(frames))/(tf.reduce_max(frames)-tf.reduce_min(frames)) # change 1
    frames = tf.cast(frames, dtype=tf.float32)
    frames = tf.transpose(frames, (0,3,2,1))

    return frames


def parse_data(example):
    # Parse Frames
    frames = tf.reshape(parse_sequence(example["frames"]), shape=(28, 61, 32, 32))
    frames = preprocess_frames(frames)
    
    # Parse Labels
    label = tf.one_hot(example["label"], depth=250)

    return frames, label


AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.TFRecordDataset(train_tfrecords)
valid_ds = tf.data.TFRecordDataset(valid_tfrecords)

trainloader = (
    train_ds
    .shuffle(configs.batch_size*4)
    .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
    .map(parse_data, num_parallel_calls=AUTOTUNE)
    .batch(configs.batch_size)
    .prefetch(AUTOTUNE)
)

validloader = (
    valid_ds
    .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
    .map(parse_data, num_parallel_calls=AUTOTUNE)
    .batch(configs.batch_size)
    .prefetch(AUTOTUNE)
)


def slowonly():
    inputs = layers.Input(shape=(28,32,32,61))
    # Stem
    x = layers.Conv3D(64, (1,7,7), 1, activation='relu')(inputs)
    # First layer
    x = layers.Conv3D(64, (1,7,7), 1)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv3D(128, (1,7,7), 1)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv3D(128, (1,7,7), 1)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv3D(256, (1,7,7), 1)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.ReLU()(x)
    
    x = layers.AveragePooling3D((1,2,2))(x)
    x = layers.GlobalAveragePooling3D()(x)
    
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(250, activation="softmax")(x)
    return models.Model(inputs, outputs)


tf.keras.backend.clear_session()
model = slowonly()
model.summary()

# total_steps = 616*configs.epochs
# decay_steps = total_steps*configs.num_steps

# cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(
#     initial_learning_rate = configs.learning_rate,
#     decay_steps = decay_steps,
#     alpha=0.1
# )

model.compile(
    tf.keras.optimizers.Adam(learning_rate=configs.learning_rate), # change 2
    tf.keras.losses.CategoricalCrossentropy(label_smoothing=configs.label_smoothing),
    metrics=["acc"]
)

run = wandb.init(
    project="kaggle-asl",
    job_type="train_poseconv3d",
    config=configs,
)

earlystopper = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    verbose=0,
    mode="auto",
    restore_best_weights=True,
)

reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=4
) # change 2

callbacks = [
    earlystopper,
    reduce_lr_on_plateau,
    WandbMetricsLogger(log_freq=2),
    WandbModelCheckpoint(
        filepath=f"model",
        save_best_only=True,
    ),
]

model.fit(
    trainloader,
    epochs=configs.epochs,
    validation_data=validloader,
    callbacks=callbacks
)

eval_loss, eval_acc = model.evaluate(validloader)
wandb.log({"eval_loss": eval_loss, "eval_acc": eval_acc})
