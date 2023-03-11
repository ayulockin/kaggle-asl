import tensorflow as tf


LIP = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17,314, 405, 321, 375, 78, 191,
    80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178,
    87, 14, 317, 402, 318, 324, 308,
]


class SeparateConvLSTMModel:
    def __init__(self, configs):
        self.configs = configs

    def get_model(self):
        inputs = tf.keras.Input((self.configs.num_frames, 543, 3), dtype=tf.float32)

        # Features
        lip_inputs = tf.gather(inputs, indices=LIP, axis=2)
        left_hand_inputs = inputs[:, :, 468:489, :]
        right_hand_inputs = inputs[:, :, 522:, :]

        lip_vector = self._conv1d_lstm_block(lip_inputs, [32, 64])
        left_hand_vector = self._conv1d_lstm_block(left_hand_inputs, [64])
        right_hand_vector = self._conv1d_lstm_block(right_hand_inputs, [64])

        vector = tf.keras.layers.Concatenate(axis=1)(
            [lip_vector, left_hand_vector, right_hand_vector]
        )
        vector = tf.keras.layers.Flatten()(vector)
        output = tf.keras.layers.Dense(250, activation="softmax")(vector)
        model = tf.keras.Model(inputs=inputs, outputs=output)

        return model
    
    def _conv1d_lstm_block(self, inputs, filters):
        vector = tf.keras.layers.ConvLSTM1D(filters=32, kernel_size=8)(inputs)
        for f in filters:
            vector = tf.keras.layers.Conv1D(filters=f, kernel_size=8)(vector)
            vector = tf.keras.layers.BatchNormalization()(vector)
            vector = tf.keras.activations.relu(vector)
            vector = tf.keras.layers.MaxPooling1D()(vector)
        vector = tf.keras.layers.Dropout(0.3)(vector)
        return vector
