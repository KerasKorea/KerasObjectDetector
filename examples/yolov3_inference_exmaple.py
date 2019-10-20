import sys, os

sys.path.insert(0, os.path.abspath('..'))
import yolk
from PIL import Image

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    image = Image.open('000000008021.jpg')
    image, shape = yolk.detector.preprocessing_image(image)

    model_path = os.path.join('..', 'yolo.h5')
    model = yolk.detector.load_model(model_path)

    output = model.predict_on_batch(np.expand_dims(image, axis=0))

    boxes, scores, labels = yolk.detector.postprocessing_output(output, shape)

    print(boxes)

if __name__ == '__main__':
    main()
