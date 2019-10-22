import sys, os
sys.path.insert(0, os.path.abspath('..'))
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import yolk
from yolk.parser import parse_args
import keras 

def main(args=None):

    if args is None:
        args = sys.argv[1:]

    args = parse_args(args)

    model = yolk.detector.load_training_model(20, args)

    train_generator = yolk.detector.get_data_generator(args)

    model.compile(
        loss=yolk.detector.get_losses(args),
        optimizer=keras.optimizers.adam(lr=args.lr, clipnorm=0.001)
    )

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=10000,
        epochs=50,
        verbose=1,
        callbacks=None,
        workers=1,
        use_multiprocessing=True,
        max_queue_size=10,
        validation_data=None
    ) 
    
if __name__ == '__main__':
    main()
