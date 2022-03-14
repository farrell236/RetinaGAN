import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import math
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from UNet2D import unet
from patient_ids import patient_ids


root_dir = '/data/datasets/retina/FGADR/Seg-set/'

























##### Tensorflow Dataloader ############################################################################################

IMG_HEIGHT = 512
IMG_WIDTH = 512

def load(image_file):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(root_dir+'Original_Images/'+image_file+'.png')
    label = tf.io.read_file(root_dir+'masks/onehot/'+image_file+'.png')
    image = tf.image.decode_png(image)
    label = tf.image.decode_png(label, 3)

    return image, label


def resize(image, label, height=IMG_HEIGHT, width=IMG_WIDTH):
    image = tf.image.resize(image, [height, width],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = tf.image.resize(label, [height, width],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image, label


def random_rotate(image, label):
    degree = tf.random.normal([]) * 360
    image = tfa.image.rotate(image, degree * math.pi / 180., interpolation='nearest')
    label = tfa.image.rotate(label, degree * math.pi / 180., interpolation='nearest')
    return image, label


def normalize(image, label):
    # normalizing the images to [-1, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.one_hot(label[..., 0], depth=7)
    return image, label


@tf.function()
def random_jitter(image, label):
    # resizing to 286 x 286 x 3
    image, label = resize(image, label)

    # Random rotate
    image, label = random_rotate(image, label)

    # randomly cropping to 256 x 256 x 3

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)

    return image, label


def load_image_train(image_file):
    image, label = load(image_file)
    image, label = random_jitter(image, label)
    image, label = normalize(image, label)

    return image, label


def load_image_test(image_file):
    image, label = load(image_file)
    image, label = resize(image, label)
    image, label = normalize(image, label)

    return image, label


train_dataset = tf.data.Dataset.from_tensor_slices(patient_ids[:1400])
train_dataset = train_dataset.shuffle(len(train_dataset))
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(4)

valid_dataset = tf.data.Dataset.from_tensor_slices(patient_ids[1400:])
valid_dataset = valid_dataset.map(load_image_test)
valid_dataset = valid_dataset.batch(2)

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.

    model = unet(num_classes=7, activation='softmax')
    # model.load_weights('ckpt/unet_fake_weights.hdf5')

    csv_logger = tf.keras.callbacks.CSVLogger(f'checkpoints/segmentation/unet_real.log')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'checkpoints/segmentation/unet_real.tf',
        monitor='val_accuracy', verbose=1,
        save_best_only=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

model.fit(train_dataset,
          validation_data=valid_dataset,
          epochs=100,
          callbacks=[checkpoint, csv_logger]
          )
