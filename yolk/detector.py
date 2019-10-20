from . import backend as M

def load_model(path, backbone='resnet50'):
    return M.load_model(path, backbone)

def preprocessing_image(image):
    return M.preprocess_image(image)

def postprocessing_output(output, shape):
    return M.postprocess_output(output, shape)
