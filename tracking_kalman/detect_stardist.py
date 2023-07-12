# @title Write a Detector function
# Import python libraries
from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt

from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D

import cv2
import numpy as np

class Detectors(object):
    """Detectors class to detect objects in video frame
    Attributes:
        None
    """
    def __init__(self):
        """Initialize variables used by Detectors class
        Args:
            None
        Return:
            None
        """

    def Detect(self, frame):
        """Detect objects in video frame using following pipeline
            - Convert captured frame from BGR to GRAY
            - Perform Background Subtraction
            - Detect edges using Canny Edge Detection
              http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
            - Retain only edges within the threshold
            - Find contours
            - Find centroids for each valid contours
        Args:
            frame: single video frame
        Return:
            centers: vector of object centroids in a frame
        """

        # Convert BGR to GRAY
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform Background Subtraction
        #fgmask = self.fgbg.apply(blurred)

        # Detect edges
        #edges = cv2.Canny(fgmask, g_thresholdValue, 190, 3)


        # Retain only edges within the threshold
        #ret, thresh = cv2.threshold(edges, g_thresholdValue, 255, 0)

        #cv2_imshow(g_thresholded)

        # Find contours
        #contours, hierarchy = cv2.findContours(thresh,
         #                                         cv2.RETR_EXTERNAL,
         #                                         cv2.CHAIN_APPROX_SIMPLE)


        # centers = []  # vector of object centroids in a frame
        # we only care about centroids with size of bug in this example
        # recommended to be tunned based on expected object size for
        # improved performance
        blob_radius_thresh = 8
        contourColor = (64,163,241)  # Orange color (BGR format)
        centroidColor = (0, 0, 255)  # Red color (BGR format)

        n_channel = 1 if img.ndim == 2 else img.shape[-1]
        axis_norm = (0,1)   # normalize channels independently
        # axis_norm = (0,1,2) # normalize channels jointly
        if n_channel > 1:
            print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

        model = StarDist2D(None, name='stardist', basedir='models')

        img = normalize(img, 1,99.8, axis=axis_norm)
        labels, details = model.predict_instances(img)
        centers = details['points']
        centers = centers[:, ::-1]
        centers = centers.astype(int)
        finalout = [];
        for center in centers:
            b = np.array([[center[0]], [center[1]]])
            center = tuple(center)
            cv2.circle(frame, center, 3, centroidColor, cv2.FILLED)
            finalout.append(np.round(b))

        # show contours of tracking objects
        # cv2.imshow('Track Bugs', frame)
        #print(len(centers))

        return finalout