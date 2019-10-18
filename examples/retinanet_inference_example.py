import sys, os
sys.path.insert(0, os.path.abspath('..'))
import yolk
from PIL import Image

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    image = np.asarray(Image.open('000000008021.jpg').convert('RGB'))
    image = image[:, :, ::-1].copy()
    image, scale = yolk.detector.preprocessing_image(image)

    model_path = os.path.join('..', 'resnet50_coco_best_v2.1.0.h5')
    model = yolk.detector.load_model(model_path)

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    print(boxes)
    
if __name__ == '__main__':
    main()
