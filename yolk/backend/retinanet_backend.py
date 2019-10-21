import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath('..'))

from PIL import Image
from keras.preprocessing import image as keras_pre_image
import keras_retinanet
from keras_retinanet import models
import keras_retinanet.utils.image as retinanet_image 

def load_model(model_path, backbone='resnet50'):
    model = models.load_model(model_path, backbone_name=backbone)
    print("type:", type(model))
    return model

def preprocess_image(image):
    image = retinanet_image.preprocess_image(image)
    image, scale = retinanet_image.resize_image(image)
    return image

def load_N_preprocess_image(
    image_path = None
):
    loaded_image = keras_pre_image.img_to_array(Image.open(image_path).convert('RGB'))
    preprocessed_image, scale = retinanet_image.resize_image(preprocess_image(loaded_image))
    return loaded_image, preprocessed_image

def postprocess_image(images, boxes, scores, labels):    
    results = []
    for idx in range(len(images)):
        results.append([])
        for box, score, label in zip(boxes[idx], scores[idx], labels[idx]): 
            results[idx].append([label + 1, score] + box.tolist()) 
    results = np.array(results) 
    return results 