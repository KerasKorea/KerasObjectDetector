from . import backend as M

def load_inference_model(path, args):
    return M.load_inference_model(path, args)

def load_training_model(num_classes, args):
    return M.load_training_model(num_classes, args)

def preprocessing_image(image, args):
    return M.preprocess_image(image)

def get_data_generator(args):
    train_generator, validation_generator = M.create_generators(args)
    return train_generator

def get_losses(args):
    return M.get_losses(args)

def postprocessing_output(output, shape):
    return M.postprocess_output(output, shape)
