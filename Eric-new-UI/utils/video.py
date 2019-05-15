import numpy as np
import cv2
import os

class Video():
    def __init__(self, path, max_frame = -1):
        if os.path.isdir(path):
            self.mode = 'images'
            self.images = []
            for filename in os.listdir(path):
                if filename.endswith('jpg') or filename.endswith('png'):
                    self.images.append(os.path.join(path, filename))   
            self.images = sorted(self.images)
            if max_frame > 0 and max_frame < len(self.images):
                self.images = self.images[:max_frame]
        elif os.path.isfile(path):
            self.mode = 'video'
            self.cap = cv2.VideoCapture(path)
            frames_num = self.cap.get(7)
            self.max_frame = max(max_frame, frames_num)
            self.frame_dict = {}
        else:
            self.mode = 'unknow'

    def maxframe(self):
        if self.mode == 'images':
            return len(self.images)
        elif self.mode == 'video':
            return self.max_frame
        else:
            return 0
    
    
    def get_frame(self, no):
        if self.mode == 'images':
            no = min(max(no, 0), len(self.images))
            image = cv2.imread(self.images[0])
            return image
        elif self.mode == 'video':
            no = min(max(no, 0), self.max_frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, no)
            ret, image = self.cap.read()
            return image
        else:
            return None

    def get_frames(self, start, end):
        image_list = []
        if self.mode == 'image':
            start = max(0, start)
            end = min(end, len(self.images))
            for i in range(start, end + 1):
                image = cv2.imread(self.images[i])
                image_list.append(image)
        elif self.mode == 'video':
            start = max(0, start)
            end = min(end, self.max_frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for i in range(start, end + 1):
                ret, image = self.cap.read()
                image_list.append(image)

        return image_list
            