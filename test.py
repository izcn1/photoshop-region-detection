import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model("./model/unet_best.h5", compile=False)

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
predict_and_visualize("./test/24870.jpg")