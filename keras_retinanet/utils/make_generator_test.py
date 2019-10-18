import keras 
import numpy as np 

from KerasObjectDetector.keras_retinanet.utils import make_generator

train_gen, val_gen = make_generator.make_generators()

sample = train_gen[0]

assert sample[0].shape[0] == sample[1][0].shape[0] 
print("generator created sucsessfully!")
