import re
import string
import random
from glob import glob
from argparse import Namespace

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

import wandb
from wandb.keras import WandbMetricsLogger

from asl.utils import id_generator, natural_keys


random_id = id_generator(size=8)
print('Experiment Id: ', random_id)


configs = Namespace(
    num_frames = 16,
    batch_size = 128,
    experiment_id = random_id,
    epochs = 20,
    use_wandb = False,
)

if configs.use_wandb:
    run = wandb.init(
        project="kaggle-asl",
        name=f"baseline-{configs.experiment_id}",
        config=configs,
        job_type="train",
    )

# Data
data_path = "data/tfrecords"
tfrecords = sorted(glob(f"{data_path}/*.tfrec"), key=natural_keys)

train_tfrecords, valid_tfrecords = tfrecords[:19], tfrecords[19:]

def parse_sequence(serialized_sequence):
    return tf.io.parse_tensor(
        serialized_sequence,
        out_type=tf.float32,
    )


def parse_tfrecord_fn(example):
    feature_description = {
        "n_frames": tf.io.FixedLenFeature([], tf.float32),
        "frames": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    
    return tf.io.parse_single_example(example, feature_description)


def true_fn(frames, n_frames):
    num_left_frames = configs.num_frames - n_frames
    left_frames = tf.zeros(shape=(num_left_frames, 543, 3))
    frames = tf.concat([frames, left_frames], 0)

    return frames


def false_fn(frames):
    frames = tf.slice(
        frames,
        begin=[0,0,0],
        size=[configs.num_frames, 543, 3]
    )

    return frames


@tf.function
def preprocess_frames(frames, n_frames):
    """This is where different preprocessing logics will be experimented."""
    # nan to num
    frames = tf.where(tf.math.is_nan(frames), 0.0, frames)
    
    # sample frames
    frames = tf.cond(
        tf.less(n_frames, configs.num_frames),
        true_fn = lambda: true_fn(frames, n_frames),
        false_fn = lambda: false_fn(frames),
    )
    
    return frames


def parse_data(example):
    # Parse Frames
    n_frames = example["n_frames"]
    frames = tf.reshape(parse_sequence(example["frames"]), shape=(n_frames, 543, 3))
    frames = preprocess_frames(frames, n_frames)
    
    # Parse Labels
    label = tf.one_hot(example["label"], depth=250)

    return frames, label

AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.TFRecordDataset(train_tfrecords)
valid_ds = tf.data.TFRecordDataset(valid_tfrecords)

trainloader = (
    train_ds
    .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
    .shuffle(1024)
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

# Model
def conv1d_lstm_block(inputs, filters):
    vector = tf.keras.layers.ConvLSTM1D(filters=32, kernel_size=8)(inputs)
    for f in filters:
        vector = tf.keras.layers.Conv1D(filters=f, kernel_size=8)(vector)
        vector = tf.keras.layers.MaxPooling1D()(vector)
    vector = tf.keras.layers.Dropout(0.3)(vector)
    return vector


LIP = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]


def get_model():
    inputs = tf.keras.Input((configs.num_frames, 543, 3), dtype=tf.float32)

    # Features
    lip_inputs = tf.gather(inputs, indices=LIP, axis=2)
    left_hand_inputs = inputs[:, :, 468:489, :]
    right_hand_inputs = inputs[:, :,522:,:]

    lip_vector = conv1d_lstm_block(lip_inputs, [32, 64])
    left_hand_vector = conv1d_lstm_block(left_hand_inputs, [64])
    right_hand_vector = conv1d_lstm_block(right_hand_inputs, [64])
    
    vector = tf.keras.layers.Concatenate(axis=1)([lip_vector, left_hand_vector, right_hand_vector])
    vector = tf.keras.layers.Flatten()(vector)
    output = tf.keras.layers.Dense(250, activation="softmax")(vector)
    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model

model = get_model()

model.compile(
    "adam",
    "binary_crossentropy",
    metrics=["acc"]
)


callbacks = []

earlystopper = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    verbose=0,
    mode="auto",
    restore_best_weights=True,
)
callbacks.append(earlystopper)

if configs.use_wandb:
    wandbmetricslogger = WandbMetricsLogger(log_freq=2)
    callbacks.append(wandbmetricslogger)

model.fit(
    trainloader,
    epochs=configs.epochs,
    validation_data=validloader,
    callbacks=callbacks,
)

# Save the model
model.save(f"models/baseline-{configs.experiment_id}")

eval_loss, eval_acc = model.evaluate(validloader)
print(f"Eval Loss: {eval_loss} | Eval Accuracy: {eval_acc}")

if configs.use_wandb:
    wandb.log({"eval_loss": eval_loss, "eval_acc": eval_acc})
