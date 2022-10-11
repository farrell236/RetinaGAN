import math
import glob
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_addons as tfa

from sklearn.model_selection import KFold



import wandb
from wandb.keras import WandbCallback
wandb.login()


fake_data_root = '/vol/medic01/users/bh1511/PyCharm_Deployment/dr_MICCAI2022/pipeline/fake_dataset/'

images = sorted(glob.glob(os.path.join(fake_data_root, 'class0/image/*.png'))) + \
         sorted(glob.glob(os.path.join(fake_data_root, 'class1/image/*.png'))) + \
         sorted(glob.glob(os.path.join(fake_data_root, 'class2/image/*.png'))) + \
         sorted(glob.glob(os.path.join(fake_data_root, 'class3/image/*.png'))) + \
         sorted(glob.glob(os.path.join(fake_data_root, 'class4/image/*.png')))

labels = [0] * 1000 + [1] * 1000 + [2] * 1000 + [3] * 1000 + [4] * 1000

df = pd.DataFrame({
    'image': images,
    'DR_grade': labels,
    })

train_val_df = df.groupby('DR_grade').sample(800, random_state=42)
# test_df = df.drop(index=train_val_df.index)


##### Tensorflow Dataloader ############################################################################################


def parse_function(filename, label):
    # Read entire contents of image
    image_string = tf.io.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.io.decode_png(image_string, channels=3)

    # Resize to 512 x 512
    # image = tf.image.resize(image, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image, label


def augmentation_fn(image, label):
    # Random left-right flip the image
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Random rotation
    degree = tf.random.normal([]) * 360
    image = tfa.image.rotate(image, degree * math.pi / 180,
                             interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Random brightness, saturation and contrast shifting
    # image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


kf = KFold(n_splits=4, random_state=42, shuffle=True)
for idx, (train_index, test_index) in enumerate(kf.split(train_val_df)):

    if idx in [0, 1, 2]: continue

    print(f'Training Fold {idx}')

    run = wandb.init(project='RetinaGAN-downstream',
                     group='classification',
                     job_type='fake',
                     name=f'fold_{idx}')

    train_images = train_val_df.iloc[train_index]['image']
    train_labels = train_val_df.iloc[train_index]['DR_grade']
    val_images = train_val_df.iloc[train_index]['image']
    val_labels = train_val_df.iloc[train_index]['DR_grade']

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(len(train_dataset))
    train_dataset = train_dataset.map(parse_function)
    train_dataset = train_dataset.map(augmentation_fn)
    train_dataset = train_dataset.batch(4)

    valid_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    valid_dataset = valid_dataset.map(parse_function)
    valid_dataset = valid_dataset.batch(2)


    ##### Network Definition ###############################################################################################

    # # Create a MirroredStrategy.
    # strategy = tf.distribute.MirroredStrategy()
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    #
    # # Open a strategy scope.
    # with strategy.scope():
    #     # Everything that creates variables should be under the strategy scope.
    #     # In general this is only model construction & `compile()`.

    model = tf.keras.Sequential([
        tf.keras.applications.InceptionResNetV2(include_top=False, weights=None, pooling='avg'),
        tf.keras.layers.Dense(5)
    ])



    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    csv_logger = tf.keras.callbacks.CSVLogger(f'CI/InceptionResNetV2_{idx}_fake.log')

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'CI/InceptionResNetV2_{idx}_fake.tf',
        monitor='val_accuracy',
        verbose=1, save_best_only=True)

    # model.fit(train_dataset, epochs=20, callbacks=[checkpoint])
    model.fit(train_dataset,
              validation_data=valid_dataset,
              epochs=50,
              callbacks=[checkpoint, csv_logger, WandbCallback()])

    del model
    del train_dataset
    del valid_dataset

    wandb.join()

########################################################################################################################
