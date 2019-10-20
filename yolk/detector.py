from . import backend as M

def load_model(path, backbone='resnet50'):
    return M.load_model(path, backbone)

def preprocessing_image(image):
    return M.preprocess_image(image)
