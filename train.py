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
from asl.data import InterpolateDataloader
from asl.model import SeparateConvLSTMModel


random_id = id_generator(size=8)
print("Experiment Id: ", random_id)


configs = Namespace(
    num_frames=16,
    batch_size=128,
    experiment_id=random_id,
    epochs=20,
    use_wandb=True,
    resizing_interpolation="nearest",
    learning_rate=1e-3,
    num_steps=1.0,
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

dataloader = InterpolateDataloader(configs)
trainloader = dataloader.get_dataloader(train_tfrecords)
validloader = dataloader.get_dataloader(valid_tfrecords, dataloader="valid")


tf.keras.backend.clear_session()
model = SeparateConvLSTMModel(configs).get_model()
model.summary()

total_steps = 585*configs.epochs
decay_steps = total_steps*configs.num_steps

cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate = configs.learning_rate,
    decay_steps = decay_steps,
    alpha=0.1
)

model.compile(
    tf.keras.optimizers.Adam(cosine_decay_scheduler),
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
