from imageio import imread
import numpy as np

import keras_ssd
from keras import models
from keras.preprocessing import image

from keras_ssd.keras_loss_function.keras_ssd_loss import SSDLoss
from keras_ssd.keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_ssd.keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_ssd.keras_layers.keras_layer_L2Normalization import L2Normalization

def load_model(model_path):
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
    model = models.load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                          'L2Normalization': L2Normalization,
                                                          'DecodeDetections': DecodeDetections,
                                                          'compute_loss': ssd_loss.compute_loss})
    return model

def preprocess_image(img_path):
    orig_images = []
    input_images = []
    img_height, img_width = 300, 300

    orig_images.append(imread(img_path))
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    input_images.append(img)
    input_images = np.array(input_images)

    return input_images