import argparse

def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """

    if parsed_args.mode == 'train' and parsed_args.dataset == None:
        raise ValueError(
            "Dataset type should be specified in training mode."
        )

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    return parsed_args




def parse_args(args):
    """
    Parse the arguments.
    backbone-model train/test dataset-type
    default:
    - backbone-model : retina
    - train/test : test
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    
    # Select which backbone-model to use
    model_parsers = parser.add_subparsers(help='Model to use in training/inference', dest='backbone_model')
    model_parsers.required = True
    
    retinanet_parser = model_parsers.add_parser('retina')
    retinanet_parser.add_argument('--mode', help='Select train or test mode', default='test', type=str)
    retinanet_parser.add_argument('--dataset', help='Arguments for specific datasets', default=None, type=str)
    retinanet_parser.add_argument('--datapath', help='Arguments for datasets direcetory', default=None, type=str)
    
    yolo_parser = model_parsers.add_parser('yolo')
    yolo_parser.add_argument('--mode', help='Select train or test mode', default='test', type=str)
    yolo_parser.add_argument('--dataset', help='Arguments for specific datasets', default=None, type=str)
    yolo_parser.add_argument('--datapath', help='Arguments for datasets direcetory', default=None, type=str)
    
    ssd_parser = model_parsers.add_parser('SSD')
    ssd_parser.add_argument('--mode', help='Select train or test mode', default='test', type=str)
    ssd_parser.add_argument('--dataset', help='Arguments for specific datasets', default=None, type=str)
    ssd_parser.add_argument('--datapath', help='Arguments for datasets direcetory', default=None, type=str)
    
    # TODO When using retinanet, there are variations of mandatory arguments according to the dataset used
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--backbone',         help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',        help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force',  help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-5)
    parser.add_argument('--snapshot-path',    help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')
    parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
    parser.add_argument('--compute-val-loss', help='Compute validation loss during training', dest='compute_val_loss', action='store_true')

    # Fit generator arguments
    parser.add_argument('--multiprocessing',  help='Use multiprocessing in fit_generator.', action='store_true')
    parser.add_argument('--workers',          help='Number of generator workers.', type=int, default=1)
    parser.add_argument('--max-queue-size',   help='Queue length for multiprocessing workers in fit_generator.', type=int, default=10)
    
    return check_args(parser.parse_args(args))
