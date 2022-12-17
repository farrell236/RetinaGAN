import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import matplotlib.pyplot as plt

# tf.random.set_seed(42)

from models.cstylegan import cStyleGAN
from models.gaugan import GauGAN
from utils import fix_pred_label, onehot_to_rgb, color_dict

BATCH_SIZE = 4
CLASS_LABEL = 4


##### LOAD Conditional StyleGAN

conditional_style_gan = cStyleGAN(start_res=4, target_res=1024)
conditional_style_gan.grow_model(256)
conditional_style_gan.load_weights(os.path.join('checkpoints/cstylegan/cstylegan_256x256.ckpt')).expect_partial()


##### LOAD GauGAN

gaugan = GauGAN(image_size=1024, num_classes=7, batch_size=BATCH_SIZE, latent_dim=512)
gaugan.load_weights('checkpoints/gaugan/gaugan_1024x1024.ckpt')

z = tf.random.normal((BATCH_SIZE, conditional_style_gan.z_dim))
w = conditional_style_gan.mapping([z, conditional_style_gan.embedding(CLASS_LABEL)])
noise = conditional_style_gan.generate_noise(batch_size=BATCH_SIZE)
labels = conditional_style_gan.call({"style_code": w, "noise": noise, "alpha": 1.0, "class_label": CLASS_LABEL})

labels = tf.keras.backend.softmax(labels)
labels = tf.cast(labels > 0.5, dtype=tf.float32)
labels = tf.image.resize(labels, (1024, 1024), method='nearest')

fixed_labels = fix_pred_label(labels)

latent_vector = tf.random.normal(shape=(BATCH_SIZE, 512), mean=0.0, stddev=2.0)
fake_image = gaugan.predict([latent_vector, fixed_labels])


##### Run Inference

for idx in range(BATCH_SIZE):
    # out = mask2img(np.argmax(labels[idx], axis=-1))
    out = onehot_to_rgb(fixed_labels[idx], color_dict)
    plt.imshow(out)
    plt.show()
    plt.imshow(fake_image[idx])
    plt.show()
    # cv2.imwrite(f'imgdump2/class_{CLASS_LABEL}_img_{idx}.png',
    #             cv2.cvtColor(fake_image[idx]*255, cv2.COLOR_RGB2BGR),
    #             [cv2.IMWRITE_PNG_COMPRESSION, 0])
