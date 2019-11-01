from imageio import imread
import numpy as np
from math import ceil

import keras_ssd
from keras import models
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger

from keras_ssd.models.keras_ssd300 import ssd_300
from keras_ssd.keras_loss_function.keras_ssd_loss import SSDLoss
from keras_ssd.keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_ssd.keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_ssd.keras_layers.keras_layer_L2Normalization import L2Normalization

from keras_ssd.ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from keras_ssd.ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from keras_ssd.data_generator.object_detection_2d_data_generator import DataGenerator
from keras_ssd.data_generator.object_detection_2d_geometric_ops import Resize
from keras_ssd.data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from keras_ssd.data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from keras_ssd.data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

from matplotlib import pyplot as plt

img_height = 300
img_width = 300
img_channels = 3
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0]
n_classes = 20 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5],
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                [1.0, 2.0, 0.5],
                [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True

def load_inference_model(model_path, args):
    model = ssd_300(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_per_layer=aspect_ratios,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=mean_color,
                    swap_channels=swap_channels)
    model.load_weights(model_path, by_name=True)
    return model

def load_training_model(n_classes, args):
    weights_path = './VGG_ILSVRC_16_layers_fc_reduced.h5'
    model = ssd_300(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_per_layer=aspect_ratios,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=mean_color,
                    swap_channels=swap_channels)
    model.load_weights(weights_path, by_name=True)
    return model

def get_losses(args):
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    return ssd_loss.compute_loss

def preprocess_image(img_path, *args):
    input_images = []
    img_height, img_width = 300, 300

    img = image.load_img(img_path, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    input_images.append(img)
    input_images = np.array(input_images)
    return input_images

def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001

def create_generators(args):
    model = args
    path = "./datasets"
    
    train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
    val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

    VOC_2007_images_dir      = path + '/VOCdevkit/VOC2007/JPEGImages/'
    VOC_2012_images_dir      = path + '/VOCdevkit/VOC2012/JPEGImages/'

    VOC_2007_annotations_dir      = path + '/VOCdevkit/VOC2007/Annotations/'
    VOC_2012_annotations_dir      = path + '/VOCdevkit/VOC2012/Annotations/'

    VOC_2007_trainval_image_set_filename = path + '/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
    VOC_2012_trainval_image_set_filename = path + '/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
    VOC_2007_test_image_set_filename     = path + '/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

    classes = ['background',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']

    train_dataset.parse_xml(images_dirs=[VOC_2007_images_dir,
                                        VOC_2012_images_dir],
                            image_set_filenames=[VOC_2007_trainval_image_set_filename,
                                                VOC_2012_trainval_image_set_filename],
                            annotations_dirs=[VOC_2007_annotations_dir,
                                            VOC_2012_annotations_dir],
                            classes=classes,
                            include_classes='all',
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)

    val_dataset.parse_xml(images_dirs=[VOC_2007_images_dir],
                        image_set_filenames=[VOC_2007_test_image_set_filename],
                        annotations_dirs=[VOC_2007_annotations_dir],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=True,
                        ret=False)

    batch_size = 32

    ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                                img_width=img_width,
                                                background=mean_color)

    convert_to_3_channels = ConvertTo3Channels()
    resize = Resize(height=img_height, width=img_width)

    predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                    model.get_layer('fc7_mbox_conf').output_shape[1:3],
                    model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                    model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                    model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                    model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

    ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                        img_width=img_width,
                                        n_classes=n_classes,
                                        predictor_sizes=predictor_sizes,
                                        scales=scales,
                                        aspect_ratios_per_layer=aspect_ratios,
                                        two_boxes_for_ar1=two_boxes_for_ar1,
                                        steps=steps,
                                        offsets=offsets,
                                        clip_boxes=clip_boxes,
                                        variances=variances,
                                        matching_type='multi',
                                        pos_iou_threshold=0.5,
                                        neg_iou_limit=0.5,
                                        normalize_coords=normalize_coords)

    train_generator = train_dataset.generate(batch_size=batch_size,
                                            shuffle=True,
                                            transformations=[ssd_data_augmentation],
                                            label_encoder=ssd_input_encoder,
                                            returns={'processed_images',
                                                    'encoded_labels'},
                                            keep_images_without_gt=False)

    val_generator = val_dataset.generate(batch_size=batch_size,
                                        shuffle=False,
                                        transformations=[convert_to_3_channels,
                                                        resize],
                                        label_encoder=ssd_input_encoder,
                                        returns={'processed_images',
                                                'encoded_labels'},
                                        keep_images_without_gt=False)

    val_dataset_size   = val_dataset.get_dataset_size()

    
    model_checkpoint = ModelCheckpoint(filepath='ssd300_pascal_07+12_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    mode='auto',
                                    period=1)

    csv_logger = CSVLogger(filename='ssd300_pascal_07+12_training_log.csv',
                        separator=',',
                        append=True)

    learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                    verbose=1)

    terminate_on_nan = TerminateOnNaN()

    callbacks = [model_checkpoint,
                csv_logger,
                learning_rate_scheduler,
                terminate_on_nan]

    val_steps = ceil(val_dataset_size/batch_size)

    return train_generator, callbacks, val_generator, val_steps

def show_result(img_path, y_pred_thresh, args):
    orig_images = [] # Store the images here.
    orig_images.append(imread(img_path))

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    classes = ['background',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']
    plt.figure(figsize=(20,12))
    plt.imshow(orig_images[0])
    current_axis = plt.gca()
    for box in y_pred_thresh[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[2] * orig_images[0].shape[1] / img_width
        ymin = box[3] * orig_images[0].shape[0] / img_height
        xmax = box[4] * orig_images[0].shape[1] / img_width
        ymax = box[5] * orig_images[0].shape[0] / img_height
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
    plt.show()