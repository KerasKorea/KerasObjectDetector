from . import backend as M

def load_inference_model(path, args):
    return M.load_inference_model(path, args)

def load_training_model(num_classes, args):
    return M.load_training_model(num_classes, args)

def preprocessing_image(image, args):
    return M.preprocess_image(image, args)

def get_data_generator(args):
    train_generator, validation_generator = M.create_generators(args)
    return train_generator

def prostprocessing_image(model_output, args):
    #bbox ,score ,~  = .prostprocessing_image(model_output)
    return 1#bbox ,score ,~ 

def get_losses(args):
    return M.get_losses(args)
     
