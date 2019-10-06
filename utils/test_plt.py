from keras.preprocessing import image 
from scipy.misc import imread
from skimage import io
import numpy as np 
from show_bbox_plt import draw_bounding_boxes_on_images

#####################################################
########## MAKE INPUTS ##############################
#####################################################

images = []
img_path = './image1.png'
img = image.load_img(img_path)
img = image.img_to_array(img)
images.append(io.imread(img_path))  

results = [ [ [0, 0.7, 0, 0, 100, 100],
              [1, 0.8, 50, 50, 150, 150] ]
]
results = np.array(results)

class_info = ['twice', 'momo', 'nayeon']

#########################################################

draw_bounding_boxes_on_images(images, results, class_info, 0.6)