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