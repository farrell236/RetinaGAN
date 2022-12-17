import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import math
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from UNet2D import unet
from patient_ids import patient_ids

# import wandb
# from wandb.keras import WandbCallback
# wandb.login()
#
# wandb.init(project='RetinaGAN-downstream',
#            group='segmentation',
#            job_type=f'fine')

root_dir = '/vol/biomedic3/bh1511/retina/FGADR/Seg-set/'

























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
    # label = tf.one_hot(label[..., 0], depth=7)  # , dtype=tf.uint8)
    label = label[..., 0][..., None]
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


def dice_coef(y_true, y_pred, num_classes=7, smooth=1e-7):
    '''
    Dice coefficient for 7 categories. Ignores background pixel label 0 and VB label 6
    Pass to model as metric during compile statement
    '''
    y_true_f = tf.keras.backend.flatten(tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)[..., 1:-1])
    y_pred_f = tf.keras.backend.flatten(y_pred[..., 1:-1])
    intersect = tf.keras.backend.sum(y_true_f * y_pred_f)
    denom = tf.keras.backend.sum(y_true_f + y_pred_f)
    return tf.keras.backend.mean((2. * intersect / (denom + smooth)))

def dice_coef_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred, alpha=0.5):
    sBCE = tf.keras.losses.SparseCategoricalCrossentropy()
    return (1 - alpha) * sBCE(y_true, y_pred) + alpha * dice_coef_loss(y_true, y_pred)


# # Create a MirroredStrategy.
# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
#
# # Open a strategy scope.
# with strategy.scope():
#     # Everything that creates variables should be under the strategy scope.
#     # In general this is only model construction & `compile()`.

# model = unet(num_classes=7, activation='softmax')
model = tf.keras.models.load_model('checkpoints/segmentation/unet_fake.tf', compile=False)

csv_logger = tf.keras.callbacks.CSVLogger(f'checkpoints/segmentation/unet_fine.log')
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'checkpoints/segmentation/unet_fine.tf',
    monitor='val_dice_coef', mode='max', verbose=1,
    save_best_only=True)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
              loss=combined_loss,
              metrics=[dice_coef])

model.fit(train_dataset,
          validation_data=valid_dataset,
          epochs=100,
          # callbacks=[checkpoint, csv_logger, WandbCallback()]
          callbacks=[checkpoint, csv_logger]
          )
