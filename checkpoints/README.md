# Model Checkpoints

Download checkpoint zips and extract them to this folder.

Required for running RetinaGAN `run_pipeline.py` and `web_demo.py`:
- [cstylegan.zip](https://xxx/research/RetinaGAN/cstylegan.zip) 
<br> Size: 561M <br> MD5: `ded1e6adf3942621a23e7b0fd8291e18`
- [gaugan.zip](https://xxx/research/RetinaGAN/gaugan.zip) 
<br> Size: 625M <br> MD5: `ed1e4d313fb6ddff416f42a7fa782d6f`

Optional checkpoints for downstream tasks:
- [segmentation.zip](https://xxx/research/RetinaGAN/segmentation.zip) 
<br> Size: 1.1G <br> MD5: `7a842f561ffe7c3060024dc822a41671`
- [classification.zip](https://xxx/research/RetinaGAN/classification.zip) 
<br> Size: 7.5G <br> MD5: `b247713b37284b975b4edec73ee12a75`

```
.
├── cstylegan
│   ├── checkpoint
│   ├── cstylegan_256x256.ckpt.data-00000-of-00001
│   └── cstylegan_256x256.ckpt.index
├── gaugan
│   ├── checkpoint
│   ├── gaugan_512x512.ckpt.data-00000-of-00001
│   └── gaugan_512x512.ckpt.index
├─ segmentation
│   ├── unet_fake.hdf5
│   ├── unet_fake.log
│   ├── unet_fake.tf
│   ├── unet_fine.log
│   ├── unet_fine.tf
│   ├── unet_real.log
│   └── unet_real.tf
└─ classification
    ├── InceptionResNetV2_0_fake.hdf5
    ├── InceptionResNetV2_0_fake.log
    ├── InceptionResNetV2_0_fake.tf
    ├── InceptionResNetV2_0_fine.log
    ├── InceptionResNetV2_0_fine.tf
    ├── InceptionResNetV2_0_real.log
    ├── InceptionResNetV2_0_real.tf
    ├── InceptionResNetV2_1_fake.hdf5
    ├── InceptionResNetV2_1_fake.log
    ├── InceptionResNetV2_1_fake.tf
    ├── InceptionResNetV2_1_fine.log
    ├── InceptionResNetV2_1_fine.tf
    ├── InceptionResNetV2_1_real.log
    ├── InceptionResNetV2_1_real.tf
    ├── InceptionResNetV2_2_fake.hdf5
    ├── InceptionResNetV2_2_fake.log
    ├── InceptionResNetV2_2_fake.tf
    ├── InceptionResNetV2_2_fine.log
    ├── InceptionResNetV2_2_fine.tf
    ├── InceptionResNetV2_2_real.log
    ├── InceptionResNetV2_2_real.tf
    ├── InceptionResNetV2_3_fake.hdf5
    ├── InceptionResNetV2_3_fake.log
    ├── InceptionResNetV2_3_fake.tf
    ├── InceptionResNetV2_3_fine.log
    ├── InceptionResNetV2_3_fine.tf
    ├── InceptionResNetV2_3_real.log
    └── InceptionResNetV2_3_real.tf
```


