from PyQt5 import QtWidgets, QtGui, uic, QtCore
import sys, os
import numpy as np
import cv2
#from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5 import QtGui
from PyQt5.QtOpenGL import *
from PyQt5.QtGui import QImage, QPixmap
sys.path.append(os.path.abspath(os.path.join('..')))
from utils import Video



class VideoThread(QtCore.QThread):

    def __init__(self, videopath, videolabel, begin = None, end = None, parent = None):
        super(VideoThread,self).__init__(parent)
        
        self.videolabel = videolabel
        self.setvideo(begin, end, videopath)

        '''
        self.begin, self.end = begin, end
        
        if begin is not None:
            self.begin = begin
        else:
            self.begin = 0
        if end is not None:
            self.end = end
        else:
            self.end = self.video.maxframe()
        '''
    
        

    def setImage(self, image):
        self.videolabel.setPixmap(QPixmap.fromImage(image))

    def run(self):
        while True:
            try:
                if self.begin is not None and self.end is not None and self.current_frame is not None:
                    self.current_frame += 1
                    if self.current_frame > self.end:
                        self.current_frame = self.begin
                    elif self.current_frame < self.begin:
                        self.current_frame = self.begin
                    frame = self.video.get_frame(self.current_frame)
                    
                    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)#在这里可以对每帧图像进行处理，
                    #p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                    self.setImage(image)
                    self.sleep(0.01)
                else:
                    self.sleep(0.5)
            except:
                pass

    def setvideo(self, begin = None, end = None, videopath = None):
        #print(begin, end, videopath)
        if videopath is not None:
            self.video = Video(videopath)
        self.begin, self.end = begin, end
        self.current_frame = self.end

