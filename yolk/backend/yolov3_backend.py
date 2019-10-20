import sys, os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import keras

from keras_yolov3.train import get_anchors, get_classes
from keras_yolov3.yolo3.model import yolo_eval
from keras_yolov3.yolo3.utils import letterbox_image


def load_model(model_path, backbone='resnet50'):
    model = keras.models.load_model(model_path)
    print("type:", type(model))
    return model

def preprocess_image(image):
    new_image_size = (image.width - (image.width % 32),
                      image.height - (image.height % 32))
    boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    return image_data, image_data.shape

def postprocess_output(output, shape):
    classes_path = '../keras_yolov3/model_data/coco_classes.txt'
    anchors_path = '../keras_yolov3/model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    num_anchors = len(anchors)

    h = output[0].shape[1]*32
    w = output[0].shape[2]*32

    input_layer = [keras.layers.Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                                             num_anchors // 3 * (num_classes + 5))) for l in range(3)]

    model_eval = keras.layers.Lambda(yolo_eval, name='yolo_eval',
                                     arguments={'anchors': anchors, 'num_classes': num_classes,
                                                'image_shape': [shape[:2]], 'max_boxes': 20,
                                                'score_threshold': 0.3, 'iou_threshold': 0.45})(input_layer)

    model = keras.models.Model(input_layer, model_eval)

    return model.predict_on_batch(output)
