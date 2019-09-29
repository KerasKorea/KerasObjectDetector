import PIL.Image as Image
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import numpy as np

def draw_bounding_box_on_image(image, box, display_class, color='red', thickness=4, is_normalized=True):
	"""

	Args:
		image:
		coordinates:
		color:
		display_class: str
	"""
	xmin, ymin, xmax, ymax = box
	
	if image is np.ndarray:
		# PIL
		image = Image.fromarray(np.uint8(image)).convert('RGB')
	
	draw = ImageDraw.Draw(image)
	im_width, im_height = image.size

	if is_normalized:
		(left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
	else:
		(left, right, top, bottom) = (xmin, xmax, ymin, ymax)
	draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)

	# font size
	#font = ImageFont.load_default()
	img_fraction = 0.05
	font_path = '/Library/Fonts/Arial.ttf'
	font = ImageFont.truetype(font_path, 24)
	#font = ImageFont.truetype(fm.findfont(fm.FontProperties(family=combo.get())), 24)
	fontsize = 1

	while font.getsize(display_class)[0] < img_fraction*image.size[0]:
		fontsize += 1
		font = ImageFont.truetype(font_path, size=fontsize)

	text_width, text_height = font.getsize(display_class)
	margin = np.ceil(0.05 * text_height)

	draw.rectangle([(left, top-text_height-2*margin), (left+text_width, top)], fill=color)
	draw.text((left, top - text_height - margin), display_class, fill='black', font=font)

	output_path = '.'
	image.save('result_image.png')


def draw_bounding_boxes_on_image(image, boxes, display_classes=[], color='red', thickness=4, is_normalized=True):
	"""

	Args:
		image:
		boxes:
		color:
		display_classes:
		thickness:
		is_normalized:
	"""
# image, box, color='red', display_class, thickness=4, is_normalized=True)
	for box, _class in zip(boxes, display_classes):
		draw_bounding_box_on_image(image, box, _class, color, thickness, is_normalized)

