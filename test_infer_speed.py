import os
import re
import json
import wandb
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import Namespace

import tensorflow as tf
print(tf.__version__)
import tensorflow_io as tfio
print(tfio.__version__)

from tensorflow.keras import layers
from tensorflow.keras import models
import tflite_runtime.interpreter as tflite


root_dir = "data/"
data_path = root_dir + "train.csv"
model_path = "models/baseline-YX3YLRKM"


def get_random_id(model_path):
    return model_path.split("-")[-1]

random_id = get_random_id(model_path)

ROWS_PER_FRAME = 543  # number of landmarks per frame

df = pd.read_csv(data_path)


def add_path(row):
    return root_dir + row.path

with open(root_dir+"sign_to_prediction_index_map.json") as f:
    label2id = json.load(f)

df["path"] = df.apply(lambda row: add_path(row), axis=1)
df["sign_encoded"] = df["sign"].apply(lambda sign: label2id[sign])


def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


configs = Namespace(
    num_frames = 16,
)

NUM_FRAMES = configs.num_frames

class DataPreprocessing(tf.Module):
    def __init__(self, num_frames=NUM_FRAMES, name=None):
        super().__init__(name=name)
        self.num_frames = num_frames
        
    def true_fn(self, frames, n_frames):
        num_left_frames = self.num_frames - n_frames
        left_frames = tf.zeros(shape=(num_left_frames, 543, 3))
        frames = tf.concat([frames, left_frames], 0)

        return frames

    def false_fn(self, frames):
        frames = tf.slice(
            frames,
            begin=[0,0,0],
            size=[self.num_frames, 543, 3]
        )

        return frames
    
    def shape_list(self, tensor):
        """
        Deal with dynamic shape in tensorflow cleanly.
        Args:
            tensor (`tf.Tensor` or `np.ndarray`): The tensor we want the shape of.
        Returns:
            `List[int]`: The shape of the tensor as a list.
        """
        if isinstance(tensor, np.ndarray):
            return list(tensor.shape)

        dynamic = tf.shape(tensor)

        if tensor.shape == tf.TensorShape(None):
            return dynamic

        static = tensor.shape.as_list()

        return [dynamic[i] if s is None else s for i, s in enumerate(static)]

    def __call__(self, frames):
        n_frames, _, _ = self.shape_list(frames)
        
        # nan to num
        frames = tf.where(tf.math.is_nan(frames), 0.0, frames)

        # sample frames
        frames = tf.cond(
            tf.less(n_frames, NUM_FRAMES),
            true_fn = lambda: self.true_fn(frames, n_frames),
            false_fn = lambda: self.false_fn(frames),
        )

        return tf.expand_dims(frames, axis=0)


model = tf.keras.models.load_model(model_path)


class TFLiteModel(tf.keras.Model):
    """
    TensorFlow Lite model that takes input tensors and applies:
        - a preprocessing model
        - the ASL model 
    """

    def __init__(self, asl_model):
        """
        Initializes the TFLiteModel with the specified feature generation model and main model.
        """
        super(TFLiteModel, self).__init__()

        # Load the feature generation and main models
        self.prep_inputs = DataPreprocessing()
        self.model = model
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')])
    def call(self, inputs):
        """
        Applies the feature generation model and main model to the input tensors.

        Args:
            inputs: Input tensor with shape [batch_size, 543, 3].

        Returns:
            A dictionary with a single key 'outputs' and corresponding output tensor.
        """
        x = self.prep_inputs(tf.cast(inputs, dtype=tf.float32))
        outputs = self.model(x)[0, :]

        # Return a dictionary with the output tensor
        return {'outputs': outputs}

tflite_keras_model = TFLiteModel(model)

keras_model_converter = tf.lite.TFLiteConverter.from_keras_model(tflite_keras_model)
tflite_model = keras_model_converter.convert()

model_lite_path = "models/model.tflite"

with open(model_lite_path, 'wb') as f:
    f.write(tflite_model)

interpreter = tflite.Interpreter(model_lite_path)
found_signatures = list(interpreter.get_signature_list().items())
print(found_signatures)

prediction_fn = interpreter.get_signature_runner("serving_default")

run = wandb.init(
    project="kaggle-asl",
    name=f"infer-{random_id}",
    config=configs,
    job_type="infer_speed",
)

for i in tqdm(range(40000)):
    output = prediction_fn(inputs=load_relevant_data_subset(df.path[i]))
    sign = np.argmax(output["outputs"])
