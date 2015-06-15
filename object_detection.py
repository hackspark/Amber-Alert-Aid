__author__ = 'jeffreytang'

import itertools
from collections import defaultdict
from scipy.spatial.distance import euclidean
import numpy as np
import cv2

class GetCarFromImage(object):
    def __init__(self, img_arr):
        self.img_arr = img_arr

    @staticmethod
    def _key_point_dist(key_points, threshold=0.8):
        """
        Takes a list of key points for an image and return a fraction of those that are close together

        :param key_points: Key points identified for an image
        :param threshold: The fraction of key points to include
        :return: A subset of the key points
        """
        key_points = [kp.pt for kp in key_points]
        key_point_pairs = itertools.combinations(key_points, 2)
        kp2dist = defaultdict(list)
        for kp1, kp2 in key_point_pairs:
            dist = euclidean(kp1, kp2)
            kp2dist[kp1].append(dist)
            kp2dist[kp2].append(dist)

        for kp, dist in kp2dist.iteritems():
            kp2dist[kp] = np.mean(dist)

        n = int(len(key_points) * threshold)
        sorted_items = sorted(kp2dist.items(), key=lambda x: x[1])[:n]
        return zip(*sorted_items)[0]

    def _object_crop_one(self, img_arr, obj_detect_class, threshold):
        """
        Accept an image as a numpy array and crop image according key points identified from object detection algo

        :param img_arr: Image as a numpy array
        :param obj_detect_class: An instantiated class with a method `detect()` to detect an object (SIFT)
        :return:
        """
        # Gray scale the image if it has not been grayscaled
        if len(img_arr.shape) > 2:
            gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_arr

        key_points = obj_detect_class.detect(gray)
        key_points = self._key_point_dist(key_points, threshold=threshold)
        x, y = zip(*key_points)
        y_min, y_max = min(y), max(y)
        x_min, x_max = min(x), max(x)
        return img_arr[y_min:y_max, x_min:x_max]

    def sift_crop(self, img_arr, contrast_threshold=0.15, threshold=0.6):
        """
        Apply sift to all images to get key points and crop those image

        :param contrast_threshold: Determine how many key points we are gonna get. Higher = Fewer key points
        :param sub_dir: The sub dir (if you want to test the transformation on 1 image)
        :param img_ind: The index of the image within the chosen sub dir
        """
        sift_detect_class = cv2.xfeatures2d.SIFT_create(contrastThreshold=contrast_threshold)
        return self._object_crop_one(img_arr, sift_detect_class, threshold)
