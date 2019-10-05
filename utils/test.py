import show_bbox
from PIL import Image

img = Image.open('image1.png')
# img = Image.open('image2.jpeg')
# show_bbox.draw_bounding_boxes_on_image(img, [(0.2, 0.2, 0.5, 0.5), (0.1, 0.1, 0.3, 0.3), (0.5, 0.5, 0.8, 0.8)], display_classes=['twice', 'nayeon', 'twice'])
show_bbox.draw_bounding_boxes_on_image(img, [(0, 0, 100, 100), (50, 50, 150, 150), (200, 200, 300, 300)], display_classes=['twice', 'nayeon', 'twice'], is_normalized=False)
