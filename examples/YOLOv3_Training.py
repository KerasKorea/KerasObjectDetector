from keras.optimizers import Adam

from keras_yolov3.train import get_anchors, get_classes, data_generator_wrapper
from keras_yolov3.yolov3_class import YOLOv3

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

classes_path = '../keras_yolov3/model_data/coco_classes.txt'
anchors_path = '../keras_yolov3/model_data/yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
num_anchors = len(anchors)

yolov3 = YOLOv3(anchors, num_classes)

model_path = '../keras_yolov3/model_data/yolo_weights.h5'
yolov3.model.load_weights(model_path, by_name=True, skip_mismatch=True)

annotation_path = '../keras_yolov3/model_data/train.txt'

with open(annotation_path) as f:
    lines = f.readlines()

num_train = len(lines)
batch_size = 32

yolov3.model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

yolov3.model.fit_generator(data_generator_wrapper(lines, batch_size, yolov3.input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train // batch_size),
                epochs=50,
                initial_epoch=0)


