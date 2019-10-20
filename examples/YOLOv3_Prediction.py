import tensorflow as tf

from PIL import Image

from keras_yolov3.train import get_anchors, get_classes
from keras_yolov3.yolov3_class import YOLOv3
from keras_yolov3.utils_img import save_img

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

classes_path = '../keras_yolov3/model_data/coco_classes.txt'
anchors_path = '../keras_yolov3/model_data/yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)

yolov3 = YOLOv3(anchors, num_classes)

model_path = '../keras_yolov3/model_data/yolo_weights.h5'
yolov3.model.load_weights(model_path)

img_path = "000000008021.jpg"
image = Image.open(img_path)

out_boxes, out_scores, out_classes = yolov3.predict_detection([image])

save_img("result.jpg", image, class_names, out_boxes[0], out_scores[0], out_classes[0])
