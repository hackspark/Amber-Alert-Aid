import numpy as np


#------------------------------
# CROP IMAGE
#------------------------------

def get_center_of_box(bounding_shape):
	x_min, x_max, y_min, y_max = bounding_shape
	return (np.mean(x_min, x_max), np.mean(y_min, y_max))

def get_rectangle_shape(center_x, center_y):
	rect_width_half = 400.0 / 2
	rect_height_half = 200.0 / 2
	x_min = center_x - rect_width_half
	x_max = center_x + rect_width_half
	y_min = center_y - rect_height_half
	y_max = center_y + rect_height_half
	return x_min, x_max, y_min, y_max

def crop_image(frame, mask, bounding_shape):
	# get bounding box center coordinates
	center_x, center_y = get_center(bounding_shape)

	# get rectangle shape
	x_min, x_max, y_min, y_max = get_rectangle_shape(center_x, center_y)

	# crop images
	cropped_frame = frame[y_min:y_max, x_min:x_max]
	cropped_mask = mask[y_min:y_max, x_min:x_max]

	return cropped_frame, cropped_mask

#------------------------------
# CENTER IMAGE
#------------------------------

# Several methods for finding the center of the car
# average location of mask pixels that are a certain brightness
# start in center of image and move center based on weighted locations of mask pixels
# clustering on mask pixels to find centroid center

from scipy import ndimage

def get_center_of_car(mask, threshold=0.0):
	if threshold:
		mask_center = ndimage.measurements.center_of_mass(mask > threshold)
	else:
		mask_center = ndimage.measurements.center_of_mass(mask)
	return mask_center
