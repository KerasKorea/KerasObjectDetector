import keras_ssd
from keras import models

from keras_ssd.keras_loss_function.keras_ssd_loss import SSDLoss
from keras_ssd.keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_ssd.keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_ssd.keras_layers.keras_layer_L2Normalization import L2Normalization

def load_model(model_path):
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
    model = models.load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                          'L2Normalization': L2Normalization,
                                                          'DecodeDetections': DecodeDetections,
                                                          'compute_loss': ssd_loss.compute_loss})
    return model
    