from PyQt5 import QtWidgets, QtGui, uic, QtCore
import sys, os, time
from PyQt5.QtWidgets import QFileDialog
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
import numpy as np
import cv2
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
import pyqtgraph.opengl as gl
import numpy as np
import pyqtgraph as pg
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5 import QtGui
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtOpenGL import *
sys.path.append(os.path.abspath(os.path.join('..')))
from track import calc_track
from plyfile import PlyData, PlyElement
from threads import VideoThread
from utils import CourtWidge, Track_Set

def clickable(widget):

    class Filter(QtCore.QObject):
    
        clicked = QtCore.pyqtSignal()
        
        def eventFilter(self, obj, event):
        
            if obj == widget:
                if event.type() == QtCore.QEvent.MouseButtonRelease:
                    if obj.rect().contains(event.pos()):
                        #print(event.pos())
                        global x, y
                        x, y = event.pos().x(), event.pos().y()
                        self.clicked.emit()
                        # The developer can opt for .emit(obj) to get the object within the slot.
                        return True
            
            return False
    
    filter = Filter(widget)
    widget.installEventFilter(filter)
    return filter.clicked

def numpy2QPixmap(img, height = None, width = None):
    shape = img.shape
    img_height, img_width = shape[:2]
    if len(shape) == 3 and shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qImage = QtGui.QImage(img.data, img_width, img_height, 3 * img_width, QtGui.QImage.Format_RGB888)
    else:
        qImage = QtGui.QImage(img.data, img_width, img_height, img_width, QtGui.QImage.Format_Grayscale8)
    
    pixmap = QtGui.QPixmap.fromImage(qImage)

    if height != None and width != None:
        pixmap = pixmap.scaled(width, height)

    return pixmap

video_cali = None
img_cali = None
objpoints = [[[0, 0, 0], [610, 0, 0], [606, 76, 0], [4, 76, 0]] , [[0, 0, 0], [610, 0, 0], [0, 155, 0], [610, 155, 0]]]
imgpoints = []
NumFrame = 1
sync_NumFrame = []
NumTrack = 1
plt_track = None

class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mywindow,self).__init__()
        uic.loadUi('./test_designer.ui', self)
        clickable(self.img_label).connect(self.mouse_click)
        self.courtwidge = CourtWidge(self.w, self.plt_widget)
        self.track_init()
        self.videothread = None
        
    
    def track_init(self):
        self.curve_list = calc_track('../track/coordinary/candidate6.txt')
        player_list = ['双方球员', '左侧球员', '右侧球员']
        for player in player_list:
            self.player_comboBox.addItem(player)

        type_list = ['所有类型']
        type_list.extend(self.curve_list[0].type_list[:-1])
        for type in type_list:
            self.type_comboBox.addItem(type)

        self.track_set = Track_Set(self.curve_list)
        self.list_candidate()
        self.player_comboBox.currentIndexChanged.connect(self.list_candidate)
        self.type_comboBox.currentIndexChanged.connect(self.list_candidate)
        
    def list_candidate(self):
        player_index, type_index = self.player_comboBox.currentIndex(), self.type_comboBox.currentIndex()
        if player_index == 0:
            player_index = None
        if type_index == 0:
            type_index = None
        else:
            type_index -= 1
        self.index_list = self.track_set.select(player_index, type_index)
        self.track_listWidget.clear()
        for index in self.index_list:
            self.track_listWidget.addItem(str(index + 1))
        
        index_list = self.track_set.select(player_index, None)
        self.courtwidge.show_statistical_result(self.track_set.get_curve_list(index_list))

    
    def videoprossing (self):
        global videoName
        #videoName="TT_left_1.MP4"
        videoName, videoType = QFileDialog.getOpenFileName(self, "选择一个视频播放吧！")

        if self.videothread is None:
            self.videothread = VideoThread(videoName, self.video_label_2, parent = self)
            if self.videothread.video.mode == 'unknow':
                self.videothread = None
            else:
                self.videothread.start()
        else:
            self.videothread.setvideo(videopath = videoName)
            if self.videothread.video.mode == 'unknow':
                self.videothread = None
        

    def setImage(self, image):
        self.video_label_2.setPixmap(QPixmap.fromImage(image))

    
    def show_badminton_court (self):
        self.courtwidge.show_badminton_court()

    
    def show_badminton_track(self):
        str_num = self.lineEdit.text()
        int_num = int(str_num)
        NumTrack = int_num
        self.courtwidge.show_badminton_track(NumTrack, self.curve_list)

            
    def click_list(self, item):
        #print('click!', item.text())
        self.courtwidge.show_badminton_track(int(item.text()), self.curve_list)
        selected_curve = self.curve_list[int(item.text()) - 1]
        type_name = selected_curve.type_list[selected_curve.type]
        text = '类型:' + type_name 
        if type_name == '下压':
            drop_speed, pass_speed = selected_curve.calc_speed()
            if pass_speed > 0:
                text += '\n过网速度: %.2f km/h'%(pass_speed / 100 / 1000 * 60 * 60)

        self.track_name.setText(text)
        if self.videothread is not None:
            begin_frame, end_frame = int(selected_curve.start * 100 + 5), int(selected_curve.end * 100 + 5)
            self.videothread.setvideo(begin = begin_frame, end = end_frame)
    
    def show_droppoint(self):
        self.courtwidge.show_droppoint(self.track_set.get_curve_list(self.index_list))
    

    def select_path(self):
        global video_cali
        path, _ = QFileDialog.getOpenFileName(self,"open file")
        video_cali = cv2.VideoCapture(path)
        self.show_cali_image()
    
    def show_cali_image(self):
        global video_cali
        global img_cali
        if video_cali == None:
            print("No video to calibration now")
        video_cali.set(cv2.CAP_PROP_POS_FRAMES, NumFrame)
        ret, img_cali = video_cali.read()
        height, width = self.img_label.height(), self.img_label.width()
        pixmap_img = numpy2QPixmap(img_cali, height, width)
        self.img_label.setPixmap(pixmap_img)
        self.video_label.setPixmap(pixmap_img)
    
    def mouse_click(self):
        global x, y, img_cali
        img_tmp = img_cali.copy()
        height, width = self.img_label.height(), self.img_label.width()
        img_height, img_width = img_cali.shape[:2]
        x, y = int(float(x) / height * img_height), int(float(y) / width * img_width)
        imgpoints.append((x,y)) 
        size = 4
        color = (0, 0, 255)
        #cv2.line(img_tmp, (x - size // 2, y), (x + size // 2, y), color, 1)
        #cv2.line(img_tmp, (x, y - size // 2), (x, y + size // 2), color, 1)
        cv2.circle(img_tmp, (x, y), 4, color, thickness = 2)
        height, width = self.img_label.height(), self.img_label.width()
        pixmap_img = numpy2QPixmap(img_tmp, height, width)
        self.img_label.setPixmap(pixmap_img)

    def keyPressEvent(self, event):
        global x, y, img_cali
        try:
            img_tmp = img_cali.copy()
            height, width = self.img_label.height(), self.img_label.width()
            img_height, img_width = img_cali.shape[:2]
            if (str(event.key()) == "65"):
                x = max(0, x - 1)
            elif (str(event.key()) == "83"):
                y = min(img_height, y + 1)
            elif (str(event.key()) == "87"):
                y = max(y - 1, 0)
            elif (str(event.key()) == "68"):
                x = min(img_width, x + 1)
            else:
                print ("you press the wrong key")
            imgpoints.pop()
            imgpoints.append((x,y))
            color = (0, 0, 255)
            cv2.circle(img_tmp, (x, y), 4, color, thickness = 2)
            height, width = self.img_label.height(), self.img_label.width()
            pixmap_img = numpy2QPixmap(img_tmp, height, width)
            self.img_label.setPixmap(pixmap_img)
        except:
            pass

    def push_calibration(self):
        global img_cali, imgpoints, objpoints
        gray = cv2.cvtColor(img_cali, cv2.COLOR_BGR2GRAY)
        imgpoints = [imgpoints[:4],imgpoints[4:]]
        objpoints = np.array(objpoints, dtype = np.float32)
        imgpoints = np.array(imgpoints, dtype = np.float32)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        print (ret)

    def push_prevframe(self):
        global video_cali, NumFrame
        video_tmp = video_cali
        if NumFrame > 1:
            NumFrame = NumFrame - 1
            video_tmp.set(cv2.CAP_PROP_POS_FRAMES, NumFrame)
            ret, img_tmp = video_tmp.read()
            height, width = self.video_label.height(), self.video_label.width()
            pixmap_img = numpy2QPixmap(img_tmp, height, width)
            self.video_label.setPixmap(pixmap_img)
        else:
            print ("It's already the first frame")
        
    def push_nextframe(self):
        global video_cali, NumFrame
        video_tmp = video_cali
        frametotal = int(video_tmp.get(7))
        if NumFrame < frametotal:
            NumFrame = NumFrame + 1
            #video_tmp.set(cv2.CAP_PROP_POS_FRAMES, NumFrame)
            ret, img_tmp = video_tmp.read()
            height, width = self.video_label.height(), self.video_label.width()
            pixmap_img = numpy2QPixmap(img_tmp, height, width)
            self.video_label.setPixmap(pixmap_img)
        else:
            print ("It's already the last frame")

    def push_thisframe(self):
        global NumFrame
        print("No.%d frame has been recorded!" %(NumFrame))
        sync_NumFrame.append(NumFrame)
        NumFrame = 1

    def change_value(self, value):
        global video_cali, NumFrame
        video_tmp = video_cali
        frametotal = int(video_tmp.get(7))
        pos = self.FrameSlider.value()
        if pos == 0:
            NumFrame = 1
        else:
            prop = frametotal/100
            NumFrame = int (pos * prop)
        video_tmp.set(cv2.CAP_PROP_POS_FRAMES, NumFrame)
        ret, img_tmp = video_tmp.read()
        height, width = self.video_label.height(), self.video_label.width()
        pixmap_img = numpy2QPixmap(img_tmp, height, width)
        self.video_label.setPixmap(pixmap_img)


## the thread for playing video
class Thread(QThread):
    changePixmap = pyqtSignal(QtGui.QImage)

    def __init__(self, parent):
        super(Thread, self).__init__(parent)

    def run(self):
        cap = cv2.VideoCapture(videoName)
        frames_num = cap.get(7)
        count = 0
        while (cap.isOpened()==True):
            count = count + 1
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)#在这里可以对每帧图像进行处理，
                #p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                p = convertToQtFormat
                self.changePixmap.emit(p)
                time.sleep(0.01)
            else:
                break


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())