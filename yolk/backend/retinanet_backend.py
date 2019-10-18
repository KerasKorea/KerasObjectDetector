import sys, os
sys.path.insert(0, os.path.abspath('..'))

import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

def load_model(model_path, backbone='resnet50'):
    model = models.load_model(model_path, backbone_name=backbone)
    print("type:", type(model))
    return model

def preprocess_image(image):
    image = keras_retinanet.utils.image.preprocess_image(image)
    image, scale = resize_image(image)
    return image, scale
