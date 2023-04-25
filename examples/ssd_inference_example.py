import sys, os
sys.path.insert(0, os.path.abspath('./'))
import numpy as np
from keras.optimizers import SGD

import yolk
from yolk.parser import parse_args

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    
    args = parse_args(args)

    # Load Sample Images
    img_path = './examples/000000008021.jpg'
    image = yolk.detector.preprocessing_image(img_path, args)

    # Generate Model including loss & sgd
    model_path = './VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.h5'
    model = yolk.detector.load_inference_model(model_path, args)
    loss = yolk.detector.get_losses(args)
    sgd = SGD(learning_rate=0.001, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(optimizer=sgd, loss=loss)


    y_pred = model.predict(image)
    # print(y_pred)

    # for clean output
    confidence_threshold = 0.5
    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_thresh[0])



    
if __name__ == '__main__':
    main()
