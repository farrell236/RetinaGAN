import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import math

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from utils import onehot_to_rgb, color_dict
from patient_ids import patient_ids


root_dir = '/vol/biomedic3/bh1511/retina/FGADR/Seg-set/'
test_ids = patient_ids[1400:]

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


def normalize(image, label):
    # normalizing the images to [-1, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.one_hot(label[..., 0], depth=7)  # , dtype=tf.uint8)
    return image, label


def load_image_test(image_file):
    image, label = load(image_file)
    image, label = resize(image, label)
    image, label = normalize(image, label)

    return image, label


test_dataset = tf.data.Dataset.from_tensor_slices(test_ids)
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(1)


##### Network Definition ###############################################################################################

# model = tf.keras.models.load_model('checkpoints/segmentation/unet_real.tf', compile=False)
# model = tf.keras.models.load_model('checkpoints/segmentation/unet_fake.tf', compile=False)
model = tf.keras.models.load_model('checkpoints/segmentation/unet_fine.tf', compile=False)


def dice_coef(y_true, y_pred, eps=1e-5):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + eps) / (np.sum(y_true_f) + np.sum(y_pred_f) + eps)


scores = []
for idx, (image, label) in enumerate(test_dataset):
    y_pred = model(image).numpy() > 0.5
    y_true = label.numpy()

    s = [
        # dice_coef(y_true[..., 0], y_pred[..., 0]),  # BG
        dice_coef(y_true[..., 1], y_pred[..., 1]),  # EX
        dice_coef(y_true[..., 2], y_pred[..., 2]),  # HE
        dice_coef(y_true[..., 3], y_pred[..., 3]),  # SE
        dice_coef(y_true[..., 4], y_pred[..., 4]),  # MA
        dice_coef(y_true[..., 5], y_pred[..., 5]),  # OD
        # dice_coef(y_true[..., 6], y_pred[..., 6]),  # VB
    ]
    print(f'{idx}: {np.round(s, 5)}')
    scores.append(s)

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(onehot_to_rgb(y_true[0], color_dict))
    # ax2.imshow(onehot_to_rgb(y_pred[0], color_dict))
    # plt.show()

print(f'Average dice per class: {np.round(np.mean(scores, axis=0), 5)}')



