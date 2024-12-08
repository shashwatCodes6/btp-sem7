import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Add, Activation, GlobalAveragePooling1D, Layer
from tensorflow.keras.initializers import TruncatedNormal
from sklearn.model_selection import train_test_split


def tcn_block(inputs, num_filters, kernel_size, dilation_rate, dropout_rate=0.2):
    x = Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="causal")(inputs)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(filters=num_filters, kernel_size=kernel_size, padding="causal")(x)
    x = Dropout(dropout_rate)(x)

    if inputs.shape[-1] != num_filters:
        inputs = Conv1D(filters=num_filters, kernel_size=1, padding="same")(inputs)

    res = Add()([inputs, x])
    res = Activation('relu')(res)
    return res


class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=2, **kwargs):
        super().__init__(**kwargs)
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        scores = tf.matmul(query, key, transpose_b=True)
        scores /= tf.sqrt(tf.cast(self.projection_dim, tf.float32))  # Scale scores
        weights = tf.nn.softmax(scores, axis=-1)  # Softmax for attention weights
        return tf.matmul(weights, value)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, timesteps, projection_dim)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.split_heads(self.query_dense(inputs), batch_size)
        key = self.split_heads(self.key_dense(inputs), batch_size)
        value = self.split_heads(self.value_dense(inputs), batch_size)
        attention = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, timesteps, num_heads, projection_dim)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        return self.combine_heads(concat_attention)



def build_tcn_attention_model(timesteps, features, num_blocks, num_filters, kernel_size):
    inputs = Input(shape=(timesteps, features))
    x = inputs
    for i in range(num_blocks):
        x = tcn_block(x, num_filters=num_filters, kernel_size=kernel_size, dilation_rate=2**i)
    x = MultiHeadSelfAttention(embed_dim=num_filters, num_heads=2)(x)
    x = GlobalAveragePooling1D()(x)
    output1 = Dense(1, activation="sigmoid", name="output1")(x)
    output2 = Dense(1, activation="sigmoid", name="output2")(x)
    return Model(inputs, [output1, output2])


def load_data_from_json(json_path):
    num_samples = 100
    max_timesteps = 60
    features = 3
    x_data = np.random.rand(num_samples, max_timesteps, features)  
    y_output1 = np.random.randint(0, 2, size=(num_samples,))
    y_output2 = np.random.randint(0, 2, size=(num_samples,))
    return x_data, y_output1, y_output2


x_data, y_output1, y_output2 = load_data_from_json("/content/angles.json")
timesteps = x_data.shape[1]
features = x_data.shape[2]

x_train, x_test, y_train_output1, y_test_output1, y_train_output2, y_test_output2 = train_test_split(
    x_data, y_output1, y_output2, test_size=0.2, random_state=42)

y_train = {"output1": y_train_output1, "output2": y_train_output2}
y_test = {"output1": y_test_output1, "output2": y_test_output2}

num_blocks = 3
num_filters = 64
kernel_size = 3

model = build_tcn_attention_model(timesteps, features, num_blocks, num_filters, kernel_size)
model.compile(
    optimizer="adam",
    loss={"output1": "binary_crossentropy", "output2": "binary_crossentropy"},
    metrics={"output1": "accuracy", "output2": "accuracy"}
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=20,
    batch_size=16
)

model.evaluate(x_test, y_test)
