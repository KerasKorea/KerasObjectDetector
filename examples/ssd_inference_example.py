import sys, os
sys.path.insert(0, os.path.abspath('./'))
import yolk

import numpy as np

def main():
    img_path = './examples/000000008021.jpg'
    image = yolk.backend.ssd_backend.preprocess_image(img_path)

    model_path = './full_VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.h5'
    model = yolk.backend.ssd_backend.load_model(model_path)

    y_pred = model.predict(image)
    print(y_pred)

    # confidence_threshold = 0.5
    # y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
    # np.set_printoptions(precision=2, suppress=True, linewidth=90)
    # print("Predicted boxes:\n")
    # print('   class   conf xmin   ymin   xmax   ymax')
    # print(y_pred_thresh[0])



    
if __name__ == '__main__':
    main()
