import tensorflow as tf


LIP = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17,314, 405, 321, 375, 78, 191,
    80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178,
    87, 14, 317, 402, 318, 324, 308,
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


class SeparateConvLSTMModel:
    def __init__(self, configs, use_attention=False):
        self.configs = configs
        self.use_attention = use_attention

    def get_model(self):
        inputs = tf.keras.Input((self.configs.num_frames, 543, 3), dtype=tf.float32)

        # Features
        lip_inputs = tf.gather(inputs, indices=LIP, axis=2)
        left_hand_inputs = inputs[:, :, 468:489, :]
        right_hand_inputs = inputs[:, :, 522:, :]
        right_eye_inputs = tf.gather(inputs, indices=RIGHT_EYE, axis=2)
        left_eye_inputs = tf.gather(inputs, indices=LEFT_EYE, axis=2)

        lip_vector = self._conv1d_lstm_block(lip_inputs, [32, 64])
        left_hand_vector = self._conv1d_lstm_block(left_hand_inputs, [64])
        right_hand_vector = self._conv1d_lstm_block(right_hand_inputs, [64])
        right_eye_vector = self._conv1d_lstm_block(right_eye_inputs, [32, 64])
        left_eye_vector = self._conv1d_lstm_block(left_eye_inputs, [32, 64])

        vector = tf.keras.layers.Concatenate(axis=1)(
            [lip_vector, left_hand_vector, right_hand_vector, right_eye_vector, left_eye_vector]
        )

        if self.use_attention:
            vector = tf.keras.layers.LayerNormalization()(vector)
            vector = tf.keras.layers.MultiHeadAttention(num_heads=3, key_dim=32)(vector, vector)
            vector = tf.keras.layers.LayerNormalization()(vector)

        vector = tf.keras.layers.Flatten()(vector)
        output = tf.keras.layers.Dense(250, activation="softmax")(vector)
        model = tf.keras.Model(inputs=inputs, outputs=output)

        return model

    def _conv1d_lstm_block(self, inputs, filters):
        x = tf.keras.layers.ConvLSTM1D(filters=32, kernel_size=8)(inputs)
        for f in filters:
            x = tf.keras.layers.Conv1D(filters=f, kernel_size=8)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.activations.relu(x)
            x = tf.keras.layers.MaxPooling1D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        return x
