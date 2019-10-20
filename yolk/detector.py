from . import backend as M

def load_N_preprocess_image(path):
    return M.load_N_preprocess_image(path)

def load_model(path, backbone='resnet50'):
    return M.load_model(path, backbone)

def preprocessing_image(image):
    return M.preprocess_image(image)
