from . import backend as M

def load_inference_model(path, args=None):
    return M.load_inference_model(path, args)

def load_training_model(num_classes, args=None):
    return M.load_training_model(num_classes, args)

def preprocess_image(image, args=None):
    return M.preprocess_image(image, args)

def get_data_generator(args):
    train_generator, validation_generator = M.create_generators(args)
    return train_generator

def postprocess_image(model_output, args=None):
    return M.postprocess_image(model_output, args)

def get_losses(args):
    return M.get_losses(args)