import sys, os
sys.path.insert(0, os.path.abspath('..'))
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import yolk
from yolk.parser import parse_args


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    
    args = parse_args(args)

    image = Image.open('./000000008021.jpg')
    image, scale = yolk.detector.preprocessing_image(image)

    model_path = os.path.join('..', 'resnet50_coco_best_v2.1.0.h5')
    model = yolk.detector.load_inference_model(model_path, args)

    model_output = model.predict_on_batch(np.expand_dims(image, axis=0))

    print(model_output)
    
    #shape = image.shape
    #boxes, scores, labels = yolk.detector.postprocessing(model_output, original_shape=image.shape, args)
    #print(boxes)
    
if __name__ == '__main__':
    main()
