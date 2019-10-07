
def create_generators(args, preprocess_image):
    """ Create generators for training and validation.

    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size'       : args.batch_size,
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'preprocess_image' : preprocess_image,
    }

    # create random transform generator for augmenting training data
    if args.random_transform:
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

    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        train_generator = CocoGenerator(
            args.coco_path,
            'train2017',
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            shuffle_groups=False,
            **common_args
        )
    elif args.dataset_type == 'pascal':
        train_generator = PascalVocGenerator(
            args.pascal_path,
            'trainval',
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'test',
            shuffle_groups=False,
            **common_args
        )
    elif args.dataset_type == 'csv':
        train_generator = CSVGenerator(
            args.annotations,
            args.classes,
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        if args.val_annotations:
            validation_generator = CSVGenerator(
                args.val_annotations,
                args.classes,
                shuffle_groups=False,
                **common_args
            )
        else:
            validation_generator = None
    elif args.dataset_type == 'oid':
        train_generator = OpenImagesGenerator(
            args.main_dir,
            subset='train',
            version=args.version,
            labels_filter=args.labels_filter,
            annotation_cache_dir=args.annotation_cache_dir,
            parent_label=args.parent_label,
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        validation_generator = OpenImagesGenerator(
            args.main_dir,
            subset='validation',
            version=args.version,
            labels_filter=args.labels_filter,
            annotation_cache_dir=args.annotation_cache_dir,
            parent_label=args.parent_label,
            shuffle_groups=False,
            **common_args
        )
    elif args.dataset_type == 'kitti':
        train_generator = KittiGenerator(
            args.kitti_path,
            subset='train',
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        validation_generator = KittiGenerator(
            args.kitti_path,
            subset='val',
            shuffle_groups=False,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator