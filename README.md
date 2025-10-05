# photoshop-region-detection
U-Net based image segmentation model trained on DIS25K for Photoshop-style region detection. Automatically extracts foreground regions, generates masks, and overlays them on input images. Colab-ready scripts included for testing and visualization.

## Features
- Photoshop-like foreground region detection
- Training with the DIS25K dataset
- Mask and overlay visualization
- Colab-ready test scripts
- Instant prediction with external image upload

## Training Result

<img width="1109" height="419" alt="image" src="https://github.com/user-attachments/assets/88e58fae-0b7d-413a-a246-7a9de9137af8" />

Accuracy rate with DIS25K data set is 89.521%

## Installation / Setup
--git clone https://github.com/izcn1/photoshop-region-detection
--cd photoshop-region-detection
--pip install -r requirements.txt
--python test.py

## Description for the test.py code
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model("./model/unet_best.h5", compile=False)  //pre-trained weight file

IMG_HEIGHT, IMG_WIDTH = 256, 256

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
    image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image_norm = image_resized / 255.0
    image_input = np.expand_dims(image_norm, axis=0)
    return image_input, image, (orig_w, orig_h)

def predict_and_visualize(image_path):
    image_input, orig_image, orig_size = preprocess_image(image_path)
    pred_mask = model.predict(image_input)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    pred_mask_resized = cv2.resize(pred_mask, orig_size)
    mask_colored = np.zeros_like(orig_image)
    mask_colored[:, :, 0] = pred_mask_resized * 255
    overlay = cv2.addWeighted(orig_image, 0.7, mask_colored, 0.3, 0)
    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.title("Orijinal Resim")
    plt.imshow(orig_image)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Tahmin Maskesi")
    plt.imshow(pred_mask_resized, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Overlay Sonuç")
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()
predict_and_visualize("./test/24870.jpg")  // test images in the test folder


## Model Files
--Download the model file from the drive link below.
#Link = https://drive.google.com/file/d/1fQZfwVnRFEr2_j2QwgtfDTYUr1oSUjzS/view?usp=drive_link
--Place the downloaded model file in the photoshop-region-detection folder
## train.ipynb
--With this file, train.ipynb, you can develop your own Photoshop detector model by training the u-net model with the data set you want.

## python test.py

<img width="1621" height="436" alt="image" src="https://github.com/user-attachments/assets/988b455a-eb01-47c3-ad5b-1ecdd4d76932" />



  


  
