"""
Copyright 2017-2018 Fizyr (https://fizyr.com)
"""
import argparse
import os
import sys
import warnings
import keras
import keras.preprocessing.image
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

from ..preprocessing.pascal_voc import PascalVocGenerator
from ..utils.anchors import make_shapes_callback
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.keras_version import check_keras_version
from ..utils.transform import random_transform_generator
from ..utils.image import random_visual_effect_generator
from ..preprocessing.download import download_pascal

def make_generators(batch_size=32, image_min_side=800, image_max_side=1333, preprocess_image=lambda x : x / 255.,
                      random_transform=True, dataset_type='voc', vesion="2012"):
    """ Create generators for training and validation.
    Args/
    
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """

    common_args = {
        'batch_size'       : batch_size,
        'image_min_side'   : image_min_side,
        'image_max_side'   : image_max_side,
        'preprocess_image' : preprocess_image,
    }

    # create random transform generator for augmenting training data
    if random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
        visual_effect_generator = random_visual_effect_generator(
            contrast_range=(0.9, 1.1),
            brightness_range=(-.1, .1),
            hue_range=(-0.05, 0.05),
            saturation_range=(0.95, 1.05)
        )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)
        visual_effect_generator = None


    # Dataset path 

    dataset_path = os.path.join(os.path.expanduser("~"), ".yolk/datasets/")

    if dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi

        if not os.path.exists(os.path.join(dataset_path, dataset_type)):
            # download_coco()
            pass

        from ..preprocessing.coco import CocoGenerator

        train_generator = CocoGenerator(
            os.path.join(dataset_path, dataset_type),
            'train2017',
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        validation_generator = CocoGenerator(
            os.path.join(dataset_path, dataset_type),
            'val2017',
            shuffle_groups=False,
            **common_args
        )

    elif dataset_type == 'voc':
    
        if not os.path.exists(os.path.join(dataset_path, dataset_type)):
            download_pascal()

        train_generator = PascalVocGenerator(
            os.path.join(dataset_path, dataset_type),
            'trainval',
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        validation_generator = PascalVocGenerator(
            os.path.join(dataset_path, dataset_type),
            'test',
            shuffle_groups=False,
            **common_args
        )
    
    else:
        raise ValueError('Invalid data type received: {}'.format(dataset_type))

    return train_generator, validation_generator


def make_generator_test():
    """
    Testing make_generator fucntion  

    usage : python make_generator.py --test True
    """
    train_gen, val_gen = make_generators()

    sample = train_gen[0]

    assert sample[0].shape[0] == sample[1][0].shape[0] 
    print("generator created sucsessfully!")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=bool, default=False, help='dataset directory on disk')
    args = parser.parse_args()

    if args.test:
        make_generator_test()

    