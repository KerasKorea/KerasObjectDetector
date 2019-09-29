import show_bbox
from PIL import Image

img = Image.open('image1.png')
show_bbox.draw_bounding_boxes_on_image(img, [(0.2, 0.2, 0.5, 0.5)], display_classes=['twice'])
