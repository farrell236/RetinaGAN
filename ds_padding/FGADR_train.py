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

from patient_ids import patient_ids

import wandb
from wandb.keras import WandbCallback
wandb.login()


root_dir = '/vol/biomedic3/bh1511/retina/FGADR/Seg-set/'

data_df = pd.read_csv(os.path.join(root_dir, 'DR_Seg_Grading_Label.csv'), names=['image', 'DR_grade'])

train_val_df = data_df[data_df['image'].isin([x + '.png' for x in patient_ids[:1400]])].copy()
# train_val_df = train_val_df.groupby('DR_grade').sample(96, random_state=42)
# train_val_df = train_val_df.groupby('DR_grade').sample(575, replace=True, random_state=42)
train_val_df['image'] = train_val_df['image'].apply(lambda x: os.path.join(root_dir, 'Original_Images', x))

test_df = data_df[data_df['image'].isin([x + '.png' for x in patient_ids[1400:]])].copy()
test_df['image'] = test_df['image'].apply(lambda x: os.path.join(root_dir, 'Original_Images', x))

a=1

# DR_grade
# 0            96 (+504)
# 1           145 (+455)
# 2           330 (+270)
# 3           575 (+25)
# 4           254 (+346)


fake_data_root = '/vol/medic01/users/bh1511/PyCharm_Deployment/dr_MICCAI2022/pipeline/fake_dataset/'

images = sorted(glob.glob(os.path.join(fake_data_root, 'class0/image/*.png')))[:504] + \
         sorted(glob.glob(os.path.join(fake_data_root, 'class1/image/*.png')))[:455] + \
         sorted(glob.glob(os.path.join(fake_data_root, 'class2/image/*.png')))[:270] + \
         sorted(glob.glob(os.path.join(fake_data_root, 'class3/image/*.png')))[:25] + \
         sorted(glob.glob(os.path.join(fake_data_root, 'class4/image/*.png')))[:346]

labels = [0] * 504 + [1] * 455 + [2] * 270 + [3] * 25 + [4] * 346

df = pd.DataFrame({
    'image': images,
    'DR_grade': labels,
    })

train_val_df = pd.concat((train_val_df, df))

##### Tensorflow Dataloader ############################################################################################


def parse_function(filename, label):
    # Read entire contents of image
    image_string = tf.io.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.io.decode_png(image_string, channels=3)

    # Resize to 512 x 512
    image = tf.image.resize(image, [512, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image, label


def augmentation_fn(image, label):
    # Random left-right flip the image
    # image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)

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


# kf = KFold(n_splits=4, random_state=42, shuffle=True)
# for idx, (train_index, test_index) in enumerate(kf.split(train_val_df)):
#
#     if idx in [1, 2, 3]: continue
#
#     print(f'Training Fold {idx}')

run = wandb.init(project='RetinaGAN-downstream',
                 group='classification',
                 job_type='dataset_padding2',
                 name=f'padding')

train_dataset = tf.data.Dataset.from_tensor_slices((train_val_df['image'], train_val_df['DR_grade']))
train_dataset = train_dataset.shuffle(len(train_dataset))
train_dataset = train_dataset.map(parse_function)
train_dataset = train_dataset.map(augmentation_fn)
train_dataset = train_dataset.batch(4)

valid_dataset = tf.data.Dataset.from_tensor_slices((test_df['image'], test_df['DR_grade']))
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
    tf.keras.layers.Dense(5, activation='softmax')
])

# model.load_weights(f'CI/InceptionResNetV2_{idx}_fake.hdf5')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

csv_logger = tf.keras.callbacks.CSVLogger(f'dataset_padding/padding.log')

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'dataset_padding/padding.tf',
    monitor='val_accuracy',
    verbose=1, save_best_only=True)

# model.fit(train_dataset, epochs=20, callbacks=[checkpoint])
model.fit(train_dataset,
          validation_data=valid_dataset,
          epochs=50,
          callbacks=[checkpoint, csv_logger, WandbCallback()])

####################################################################################################################
