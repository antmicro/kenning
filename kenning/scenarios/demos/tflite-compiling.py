"""
Test demo for verifying the TFLite compilation process.
"""

from kenning.datasets.pet_dataset import PetDataset
import tensorflow as tf
import random


dataset = PetDataset('./build/pet-dataset')


mean, std = dataset.get_input_mean_std()


def preprocess_input(path):
    global mean
    global std
    data = tf.io.read_file(path)
    img = tf.io.decode_jpeg(data, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img /= 255.0
    img = (img - mean) / std
    return tf.expand_dims(img, axis=0)


def generator():
    lst = [X for X in dataset.get_data_unloaded()[0]][::10]
    random.shuffle(lst)
    for X in lst:
        X = preprocess_input(X)
        yield [X]


model = tf.keras.models.load_model('./build/larger-net-2.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_opts = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

converter.representative_dataset = generator
tflite_model = converter.convert()

with open('quantized.tflite', 'wb') as f:
    f.write(tflite_model)
