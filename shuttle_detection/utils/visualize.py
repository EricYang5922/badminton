import numpy as np 
import cv2

def label_points(img, points):
    result_img = img.copy()
    for point in points:
        cv2.circle(result_img, (point[1], point[0]), 3, (255, 0, 0), -1)
    return result_img


