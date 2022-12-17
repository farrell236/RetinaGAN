import math
import os

import tqdm

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_addons as tfa

from patient_ids import patient_ids


root_dir = '/vol/biomedic3/bh1511/retina/FGADR/Seg-set/'

data_df = pd.read_csv(os.path.join(root_dir, 'DR_Seg_Grading_Label.csv'), names=['image', 'DR_grade'])

train_df = data_df[data_df['image'].isin([x + '.png' for x in patient_ids[:1400]])]
test_df = data_df[data_df['image'].isin([x + '.png' for x in patient_ids[1400:]])]


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
    # label = tf.one_hot(label, depth=5)

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


# On Local Storage:
train_df['image'] = root_dir + 'Original_Images/' + train_df['image']
test_df['image'] = root_dir + 'Original_Images/' + test_df['image']

train_dataset = tf.data.Dataset.from_tensor_slices((train_df['image'], train_df['DR_grade']))
train_dataset = train_dataset.shuffle(len(train_dataset))
train_dataset = train_dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(augmentation_fn, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(16)

valid_dataset = tf.data.Dataset.from_tensor_slices((test_df['image'], test_df['DR_grade']))
valid_dataset = valid_dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(8)


##### Network Definition ###############################################################################################

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, jaccard_score, confusion_matrix, ConfusionMatrixDisplay


model = tf.keras.models.load_model('dataset_padding/base.tf', compile=False)
# model = tf.keras.models.load_model('dataset_padding/undersample.tf', compile=False)
# model = tf.keras.models.load_model('dataset_padding/oversample.tf', compile=False)
# model = tf.keras.models.load_model('dataset_padding/padding.tf', compile=False)

a=1

y_pred_all = []
y_true_all = []
for image, label in tqdm.tqdm(valid_dataset):
    y_pred_all.append(model(image))
    y_true_all.append(label)
y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)
y_pred_all = np.argmax(y_pred_all, axis=-1)

print(accuracy_score(y_true_all, y_pred_all))
print(precision_score(y_true_all, y_pred_all, average='weighted'))
print(recall_score(y_true_all, y_pred_all, average='weighted'))
print(f1_score(y_true_all, y_pred_all, average='weighted'))
print(jaccard_score(y_true_all, y_pred_all, average='weighted'))

cm = confusion_matrix(y_true_all, y_pred_all)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
# cm_display.plot().figure_.savefig('padding.png')

# classification_report(y_true_all, y_pred_all)

########################################################################################################################
