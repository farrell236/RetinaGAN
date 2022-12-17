import math
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import tensorflow_addons as tfa

from matplotlib import pyplot as plt
from patient_ids import patient_ids
from models.gaugan import GauGAN

BATCH_SIZE = 4
IMG_HEIGHT = IMG_WIDTH = 1024
NUM_CLASSES = 7
AUTOTUNE = tf.data.AUTOTUNE


root_dir = '/vol/biomedic3/bh1511/retina/FGADR/Seg-set/'


def load(image_file):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(root_dir+'Original_Images/'+image_file+'.png')
    labels = tf.io.read_file(root_dir+'masks/onehot/'+image_file+'.png')
    segmentation_map = tf.io.read_file(root_dir+'masks/rgb/'+image_file+'.png')
    image = tf.image.decode_png(image, 3)
    labels = tf.image.decode_png(labels, 3)
    segmentation_map = tf.image.decode_png(segmentation_map, 3)

    return segmentation_map, image, labels


def resize(segmentation_map, image, labels, height, width):
    segmentation_map = tf.image.resize(segmentation_map, [height, width],
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.resize(image, [height, width],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    labels = tf.image.resize(labels, [height, width],
                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return segmentation_map, image, labels


def random_rotate(segmentation_map, image, labels):
    degree = tf.random.normal([]) * 360
    segmentation_map = tfa.image.rotate(segmentation_map, degree * math.pi / 180., interpolation='nearest')
    image = tfa.image.rotate(image, degree * math.pi / 180., interpolation='nearest')
    labels = tfa.image.rotate(labels, degree * math.pi / 180., interpolation='nearest')
    return segmentation_map, image, labels


def random_crop(segmentation_map, image, labels):
    stacked_image = tf.stack([segmentation_map, image, labels], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[3, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1], cropped_image[2]


def normalize(segmentation_map, image, labels):
    # normalizing the images to [-1, 1]
    labels = tf.one_hot(labels[..., 0], depth=7)  # , dtype=tf.uint8)
    image = tf.image.convert_image_dtype(image, tf.float32)
    segmentation_map = tf.image.convert_image_dtype(segmentation_map, tf.float32)
    return segmentation_map, image, labels


@tf.function()
def random_jitter(segmentation_map, image, labels):
    # resizing to 286 x 286 x 3
    segmentation_map, image, labels = resize(segmentation_map, image, labels,
                                             int(IMG_HEIGHT*1.1), int(IMG_WIDTH*1.1))

    # Random rotate
    segmentation_map, image, labels = random_rotate(segmentation_map, image, labels)

    # randomly cropping to 256 x 256 x 3
    segmentation_map, image, labels = random_crop(segmentation_map, image, labels)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        segmentation_map = tf.image.flip_left_right(segmentation_map)
        image = tf.image.flip_left_right(image)
        labels = tf.image.flip_left_right(labels)

    return segmentation_map, image, labels


def load_image_train(image_file):
    segmentation_map, image, labels = load(image_file)
    segmentation_map, image, labels = random_jitter(segmentation_map, image, labels)
    segmentation_map, image, labels = normalize(segmentation_map, image, labels)

    return segmentation_map, image, labels


def load_image_test(image_file):
    segmentation_map, image, labels = load(image_file)
    segmentation_map, image, labels = resize(segmentation_map, image, labels,
                                             IMG_HEIGHT, IMG_WIDTH)
    segmentation_map, image, labels = normalize(segmentation_map, image, labels)

    return segmentation_map, image, labels


train_dataset = tf.data.Dataset.from_tensor_slices(patient_ids[:1400])
train_dataset = train_dataset.shuffle(len(train_dataset))
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

test_dataset = tf.data.Dataset.from_tensor_slices(patient_ids[1400:])
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)


class GanMonitor(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset, n_samples, epoch_interval=5):
        self.val_images = next(iter(val_dataset))
        self.n_samples = n_samples
        self.epoch_interval = epoch_interval

    def infer(self):
        latent_vector = tf.random.normal(
            shape=(self.model.batch_size, self.model.latent_dim), mean=0.0, stddev=2.0
        )
        return self.model.predict([latent_vector, self.val_images[2]])

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epoch_interval == 0:
            generated_images = self.infer()
            for _ in range(self.n_samples):
                grid_row = min(generated_images.shape[0], 3)
                f, axarr = plt.subplots(grid_row, 3, figsize=(18, grid_row * 6))
                for row in range(grid_row):
                    ax = axarr if grid_row == 1 else axarr[row]
                    ax[0].imshow(self.val_images[0][row])
                    ax[0].axis("off")
                    ax[0].set_title("Mask", fontsize=20)
                    ax[1].imshow(self.val_images[1][row])
                    ax[1].axis("off")
                    ax[1].set_title("Ground Truth", fontsize=20)
                    ax[2].imshow(generated_images[row])
                    ax[2].axis("off")
                    ax[2].set_title("Generated", fontsize=20)
                plt.show()


ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    f'checkpoints/gaugan_bs8/gaugan_1024x1024.ckpt',
    save_weights_only=True,
    verbose=0,
)

gaugan = GauGAN(IMG_HEIGHT, NUM_CLASSES, BATCH_SIZE, latent_dim=512)
gaugan.compile()
gaugan.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=100,
    callbacks=[GanMonitor(test_dataset, BATCH_SIZE), ckpt_cb],
)
