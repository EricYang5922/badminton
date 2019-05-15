from PyQt5 import QtWidgets, QtGui, uic, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import random

class MplWidget(QtWidgets.QWidget):
    
    def __init__(self, parent = None):

        super(MplWidget, self).__init__(parent)
        
        self.canvas = FigureCanvas(Figure())
        
        vertical_layout = QtWidgets.QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)

    
    def plot(self):
        '''
        fs = 500
        f = random.randint(1, 100)
        ts = 1/fs
        length_of_signal = 100
        t = np.linspace(0,1,length_of_signal)
        
        cosinus_signal = np.cos(2*np.pi*f*t)
        sinus_signal = np.sin(2*np.pi*f*t)

        self.canvas.axes.clear()
        self.canvas.axes.plot(t, cosinus_signal)
        self.canvas.axes.plot(t, sinus_signal)
        self.canvas.axes.legend(('cosinus', 'sinus'),loc='upper right')
        self.canvas.axes.set_title('Cosinus - Sinus Signal')
        '''
        
        labels=['China','Swiss','USA','UK','Laos','Spain']
        X=[222,42,455,664,454,334]  
        
        #fig = plt.figure()
        self.canvas.axes.pie(X,labels=labels,autopct='%1.2f%%') #画饼图（数据，数据对应的标签，百分数保留两位小数点）
        self.canvas.draw()

        
        '''
        vertical_layout = QtWidgets.QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)
        '''