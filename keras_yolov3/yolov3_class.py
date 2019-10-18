from keras.layers import Input, Lambda
from keras.models import Model

from keras_yolov3.yolo3.model import yolo_eval, yolo_loss, yolo_body
from keras_yolov3.yolo3.utils import letterbox_image

import numpy as np


class YOLOv3:
    def __init__(self, anchors, num_classes=80, **kwargs):
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.input_shape = (416, 416)

        self.model_body = self._build_body()
        self.model = self._build_train()

    def _build_body(self):
        image_input = Input(shape=(None, None, 3))

        return yolo_body(image_input, self.num_anchors // 3, self.num_classes)

    def _build_train(self):
        h, w = self.input_shape

        y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                               self.num_anchors//3, self.num_classes + 5)) for l in range(3)]

        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                            arguments={'anchors': self.anchors, 'num_classes': self.num_classes,
                                       'ignore_thresh': 0.5})([*self.model_body.output, *y_true])

        return Model([self.model_body.input, *y_true], model_loss)

    def _preprocess_input(self, inputs):
        for i in range(len(inputs)):
            boxed_image = letterbox_image(inputs[i], self.input_shape)
            image_data = np.array(boxed_image, dtype='float32')
            image_data /= 255.
            inputs[i] = image_data

        return inputs

    def predict_detection(self, inputs):
        images_shape_org = []
        for image in inputs:
           images_shape_org.append([image.size[1], image.size[0]])

        inputs = np.array(self._preprocess_input(inputs), dtype='float32')

        model_eval = Lambda(yolo_eval, name='yolo_eval',
                            arguments={'anchors': self.anchors, 'num_classes': self.num_classes,
                                       'image_shape': images_shape_org, 'max_boxes': 20,
                                       'score_threshold': 0.3, 'iou_threshold': 0.45})(self.model_body.output)

        model = Model(self.model_body.input, model_eval)

        return model.predict_on_batch(inputs)
