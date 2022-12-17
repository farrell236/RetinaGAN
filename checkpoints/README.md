# Model Checkpoints

Download checkpoint zips and extract them to this folder.

Required for running RetinaGAN `run_pipeline.py` and `web_demo.py`:
- [cstylegan.zip](https://www.doc.ic.ac.uk/~bh1511/research/RetinaGAN/cstylegan.zip) 
<br> Size: 561M <br> MD5: `ded1e6adf3942621a23e7b0fd8291e18`
- [gaugan.zip](https://www.doc.ic.ac.uk/~bh1511/research/RetinaGAN/gaugan.zip) 
<br> Size: 1.6G <br> MD5: `40199790d9676934af6d4a0039954b85`

Optional checkpoints for downstream tasks:
- [dataset_padding.zip](https://www.doc.ic.ac.uk/~bh1511/research/RetinaGAN/dataset_padding.zip) 
<br> Size: 2.3G <br> MD5: `9a0c4a4c0e5557426d1e24a9303793b4`
- [segmentation.zip](https://www.doc.ic.ac.uk/~bh1511/research/RetinaGAN/segmentation.zip) 
<br> Size: 974M <br> MD5: `4fbf7332e07a6723ca42767ffd91f07a`
- [classification.zip](https://www.doc.ic.ac.uk/~bh1511/research/RetinaGAN/classification.zip) 
<br> Size: 6.8G <br> MD5: `c33ac67fc4f7769e8d80a8615749b7db`

```
.
├── README.md
├── cstylegan
│   ├── checkpoint
│   ├── cstylegan_256x256.ckpt.data-00000-of-00001
│   └── cstylegan_256x256.ckpt.index
├── gaugan
│   ├── checkpoint
│   ├── gaugan_1024x1024.ckpt.data-00000-of-00001
│   └── gaugan_1024x1024.ckpt.index
├── dataset_padding
│   ├── base.log
│   ├── base.tf
│   ├── oversample.log
│   ├── oversample.tf
│   ├── padding.log
│   ├── padding.tf
│   ├── undersample.log
│   └── undersample.tf
├── segmentation
│   ├── unet_fake.log
│   ├── unet_fake.tf
│   ├── unet_fine.log
│   ├── unet_fine.tf
│   ├── unet_real.log
│   └── unet_real.tf
└── classification
    ├── InceptionResNetV2_0_fake.log
    ├── InceptionResNetV2_0_fake.tf
    ├── InceptionResNetV2_0_fine.log
    ├── InceptionResNetV2_0_fine.tf
    ├── InceptionResNetV2_0_real.log
    ├── InceptionResNetV2_0_real.tf
    ├── InceptionResNetV2_1_fake.log
    ├── InceptionResNetV2_1_fake.tf
    ├── InceptionResNetV2_1_fine.log
    ├── InceptionResNetV2_1_fine.tf
    ├── InceptionResNetV2_1_real.log
    ├── InceptionResNetV2_1_real.tf
    ├── InceptionResNetV2_2_fake.log
    ├── InceptionResNetV2_2_fake.tf
    ├── InceptionResNetV2_2_fine.log
    ├── InceptionResNetV2_2_fine.tf
    ├── InceptionResNetV2_2_real.log
    ├── InceptionResNetV2_2_real.tf
    ├── InceptionResNetV2_3_fake.log
    ├── InceptionResNetV2_3_fake.tf
    ├── InceptionResNetV2_3_fine.log
    ├── InceptionResNetV2_3_fine.tf
    ├── InceptionResNetV2_3_real.log
    └── InceptionResNetV2_3_real.tf
```


