import cv2

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from models.cstylegan import cStyleGAN
from models.gaugan import GauGAN
from utils import fix_pred_label, onehot_to_rgb, rgb_to_onehot, color_dict

from skimage import io


@st.cache(allow_output_mutation=True)
def load_cstylegan():
    conditional_style_gan = cStyleGAN(start_res=4, target_res=1024)
    conditional_style_gan.grow_model(256)
    conditional_style_gan.load_weights('checkpoints/cstylegan/cstylegan_256x256.ckpt').expect_partial()
    print('Conditional StyleGAN Model Loaded!')
    return conditional_style_gan


@st.cache(allow_output_mutation=True)
def load_gaugan(batch_size):
    gaugan = GauGAN(image_size=1024, num_classes=7, batch_size=batch_size, latent_dim=512)
    gaugan.load_weights('checkpoints/gaugan/gaugan_1024x1024.ckpt').expect_partial()
    print('GauGAN Model Loaded!')
    return gaugan


def set_seed():
    tf.random.set_seed(seed=st.session_state.seed)


def main():

    st.title('RetinaGAN')

    st.sidebar.columns([1, 5, 1])[1].image(cv2.cvtColor(cv2.imread('assets/sample.jpeg'), cv2.COLOR_BGR2RGB))

    st.sidebar.title('Menu')
    options = st.sidebar.selectbox('Select Option:', ('About', 'Random', 'Upload your own', 'Retina Template'))

    if options == 'About':
        st.write('Online Demo for **High-Fidelity Diabetic Retina Fundus Image Synthesis from Freestyle Lesion Maps**.')

        st.write('''        
        Paper: LINK_TBD
        
        Github: http://github.com/farrell236/RetinaGAN
        
        👈 Select an Option From the drop down menu

        ---
        ''')

        st.write('''
        RetinaGAN a two-step process for generating photo-realistic retinal 
        Fundus images based on artificially generated or free-hand drawn semantic lesion maps.
        ''')

        st.columns([1, 5, 1])[1].image(cv2.cvtColor(cv2.imread('assets/RetinaGAN_pipeline.png'), cv2.COLOR_BGR2RGB),
                                       caption='RetinaGAN Pipeline')

        st.write('''
        StyleGAN is modified to be conditional in to synthesize pathological lesion maps 
        based on a specified DR grade (i.e., grades 0 to 4). The DR Grades are defined by the 
        International Clinical Diabetic Retinopathy (ICDR) disease severity scale; 
        no apparent retinopathy, {mild, moderate, severe} Non-Proliferative Diabetic Retinopathy (NPDR), 
        and Proliferative Diabetic Retinopathy (PDR). The output of the network is a binary image with 
        seven channels instead of class colors to avoid ambiguity.
        ''')

        st.columns([1, 5, 1])[1].image(cv2.cvtColor(cv2.imread('assets/cStyleGAN.png'), cv2.COLOR_BGR2RGB),
                                       caption='Conditional StyleGAN Model')

        st.write('''
        The generated label maps are then passed through GauGAN, an image-to-image translation network, 
        to turn them into photo-realistic retina fundus images. The input to the network are one-hot 
        encoded labels.
        ''')

        st.columns([1, 5, 1])[1].image(cv2.cvtColor(cv2.imread('assets/GauGAN.png'), cv2.COLOR_BGR2RGB),
                                       caption='GauGAN Model')


    elif options == 'Random':

        st.session_state.seed = st.sidebar.number_input('Sampling Seed:', value=42, on_change=set_seed)

        ## Load Models
        conditional_style_gan = load_cstylegan()
        gaugan = load_gaugan(4)

        for idx, col in enumerate(st.columns(5)):

            z = tf.random.normal((1, conditional_style_gan.z_dim))
            w = conditional_style_gan.mapping([z, conditional_style_gan.embedding(idx)])
            noise = conditional_style_gan.generate_noise(batch_size=1)
            labels = conditional_style_gan.call({"style_code": w, "noise": noise, "alpha": 1.0, "class_label": idx})

            labels = tf.keras.backend.softmax(labels)
            labels = tf.cast(labels > 0.5, dtype=tf.float32)
            labels = tf.image.resize(labels, (1024, 1024), method='nearest')

            fixed_labels = fix_pred_label(labels)
            fixed_labels = tf.tile(fixed_labels, (4, 1, 1, 1))

            latent_vector = tf.random.normal(shape=(4, 512), mean=0.0, stddev=2.0)
            fake_image = gaugan.predict([latent_vector, fixed_labels])

            with col:
                st.text(f'DR Grade {idx}')
                st.image(onehot_to_rgb(fixed_labels[0], color_dict), output_format='PNG')
                for im in fake_image:
                    st.image(im)

        # Run again?
        st.button('Regenerate Images')

    elif options == 'Upload your own':

        st.session_state.seed = st.sidebar.number_input('Sampling Seed:', value=42, on_change=set_seed)

        st.sidebar.info('PRIVACY POLICY: Uploaded images are never stored on disk.')

        ## Load Models
        gaugan = load_gaugan(1)

        uploaded_file = st.file_uploader('Choose an image...', type=('png'))

        if uploaded_file:
            col1, col2 = st.columns(2)

            # Read input image with size [H, W, 3] and range (0, 255)
            img_array = io.imread(uploaded_file)[..., 0:3]

            # Test for valid mask
            test_colours = np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0)
            if not all([tuple(x) in color_dict.values() for x in test_colours]):
                st.info('Mask Contains invalid Class Colours')
                return

            # Resize image with padding to [1024, 1024, 3]
            img_array = tf.image.resize_with_pad(img_array, 1024, 1024, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            # Display input image
            with col1:
                st.image(img_array.numpy(), caption='Uploaded Image')

            img_label = rgb_to_onehot(img_array.numpy(), color_dict)[None, ...]
            latent_vector = tf.random.normal(shape=(1, 512), mean=0.0, stddev=2.0)
            fake_image = gaugan.predict([latent_vector, img_label])[0]

            with col2:
                st.image(fake_image, caption='Generated Image')

        # Run again?
        st.button('Regenerate Image')

    elif options == 'Retina Template':

        st.header('Template')

        st.write('Download the Retina Template image below. '
                 'Using an image editor of your choice, paint lesions '
                 'into the Vitreous Body and upload it to the model. '
                 'NB: Images must be stored as lossless PNGs')

        template = np.uint8(cv2.circle(np.zeros((1024, 1024, 3)), [512, 512], 512, (255, 255, 255), -1))
        st.columns([1, 5, 1])[1].image(template, use_column_width=True, output_format='PNG')

        st.header('Class Colours')
        cols = st.columns(7)
        for idx, cls in enumerate(color_dict):
            with cols[idx]:
                st.image(image=np.tile(color_dict[cls], (32, 32, 1)),
                         caption=cls,
                         output_format='PNG')
                # st.caption(color_dict[cls])


        data = {'Class Name': [
                    'Background',
                    'Hard Exudate',
                    'Hemohedge',
                    'Soft Exudate',
                    'Micro Aneurysms',
                    'Optical Disc',
                    'Vitreous Body'],
                'RGB Colour': [
                    str(color_dict[0]),  # BG
                    str(color_dict[1]),  # EX
                    str(color_dict[2]),  # HE
                    str(color_dict[3]),  # SE
                    str(color_dict[4]),  # MA
                    str(color_dict[5]),  # OD
                    str(color_dict[6])]  # VB
                }

        st.table(data)


if __name__ == '__main__':

    # tf.config.set_visible_devices([], 'GPU')

    main()


