from PyQt5 import QtWidgets, QtGui, uic, QtCore
import sys, os
import numpy as np
import cv2
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5 import QtGui
from PyQt5.QtOpenGL import *
sys.path.append(os.path.abspath(os.path.join('..')))
from plyfile import PlyData, PlyElement
import copy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pylab import mpl


class CourtWidge():
    def __init__(self, gl_widge, plt_widget):
        mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
        mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
        self.w = gl_widge
        self.plt_w = plt_widget
        self.plt_w.plot()
        self.plt_track = None
        self.badminton_model = self.get_badminton_model()
        self.badminton_item_list, self.track_item_list, self.point_item_list = [], [], []
        self.show_badminton_court()

    def get_badminton_model(self):
        badminton = PlyData.read('../calibration_eric/badminton.ply')
        _vertex = []
        for data in badminton['vertex']:
            _vertex.append((data[0], data[1], data[2]))
        _vertex = np.array(_vertex)

        _face = []
        _facecolor = []
        for data in badminton['face']:
            _face.append((data[0][0], data[0][1], data[0][2]))
            _facecolor.append((data[1],data[2],data[3],data[4]))

        _face = np.array(_face)  
        _facecolor = np.array(_facecolor)
        meshdata = gl.MeshData(vertexes= _vertex, faces=_face, faceColors=_facecolor)
        #meshitem = gl.GLMeshItem(meshdata = meshdata)
        return meshdata 

    def show_badminton_court(self):
        w = self.w
        #w = gl.GLViewWidget()
        w.opts['distance'] = 1500
        w.show()
        #w.setWindowTitle('badminton court')

        z = np.zeros((2,2))
        x = np.linspace(-700, 700, 2)
        y = np.linspace(-350, 350, 2)
        colors = np.zeros((2, 2, 3), dtype=float)
        colors[...,0] = 0
        colors[...,2] = 0
        colors[...,1] = 0.5
        ground = gl.GLSurfacePlotItem(x, y, z=z, colors=colors.reshape(2*2,3), shader='shaded', smooth=False)
        w.addItem(ground)

        
        def draw_line(x0, x1, y0, y1, z0 = 1, colors = None):
            if colors is None:
                colors = np.ones((4, 3), dtype=float)
            z = np.full((2,2), z0)
            x = np.linspace(x0, x1, 2)
            y = np.linspace(y0, y1, 2)
            l = gl.GLSurfacePlotItem(x, y, z=z, colors=colors, shader='shaded', smooth=False)
            w.addItem(l)

        draw_line(-670, -666, -305, 305)
        draw_line(666, 670, -305, 305)
        draw_line(-670, 670, 301, 305)
        draw_line(-670, 670, -305, -301)
        draw_line(-670, 670, -259, -255)
        draw_line(-670, 670, 255, 259)
        draw_line(-594, -590, -305, 305)
        draw_line(590, 594, -305, 305)
        draw_line(-202, -198, -305, 305)
        draw_line(198, 202, -305, 305)
        draw_line(-670, -198, -2, 2)
        draw_line(198, 670, -2, 2)

        #pole1
        x = np.linspace(-2, 2, 2)
        y = np.linspace(305,305,2)
        z = np.array([(1,160),(1,160)])
        colors = np.ones((2, 2, 3), dtype=float)
        l = gl.GLSurfacePlotItem(x, y, z=z, colors=colors.reshape(2*2,3), shader='shaded', smooth=False)
        w.addItem(l)
        x = np.linspace(-2, 2, 2)
        y = np.linspace(301,301,2)
        z = np.array([(1,160),(1,160)])
        colors = np.ones((2, 2, 3), dtype=float)
        l = gl.GLSurfacePlotItem(x, y, z=z, colors=colors.reshape(2*2,3), shader='shaded', smooth=False)
        w.addItem(l)
        x = np.linspace(-2, -2, 2)
        y = np.linspace(301,305,2)
        z = np.array([(1,1),(160,160)])
        colors = np.ones((2, 2, 3), dtype=float)
        l = gl.GLSurfacePlotItem(x, y, z=z, colors=colors.reshape(2*2,3), shader='shaded', smooth=False)
        w.addItem(l)
        x = np.linspace(2, 2, 2)
        y = np.linspace(301,305,2)
        z = np.array([(1,1),(160,160)])
        colors = np.ones((2, 2, 3), dtype=float)
        l = gl.GLSurfacePlotItem(x, y, z=z, colors=colors.reshape(2*2,3), shader='shaded', smooth=False)
        w.addItem(l)
        #pole2
        x = np.linspace(-2, 2, 2)
        y = np.linspace(-305,-305,2)
        z = np.array([(1,160),(1,160)])
        colors = np.ones((2, 2, 3), dtype=float)
        l = gl.GLSurfacePlotItem(x, y, z=z, colors=colors.reshape(2*2,3), shader='shaded', smooth=False)
        w.addItem(l)
        x = np.linspace(-2, 2, 2)
        y = np.linspace(-301,-301,2)
        z = np.array([(1,160),(1,160)])
        colors = np.ones((2, 2, 3), dtype=float)
        l = gl.GLSurfacePlotItem(x, y, z=z, colors=colors.reshape(2*2,3), shader='shaded', smooth=False)
        w.addItem(l)
        x = np.linspace(-2, -2, 2)
        y = np.linspace(-305,-301,2)
        z = np.array([(1,1),(160,160)])
        colors = np.ones((2, 2, 3), dtype=float)
        l = gl.GLSurfacePlotItem(x, y, z=z, colors=colors.reshape(2*2,3), shader='shaded', smooth=False)
        w.addItem(l)
        x = np.linspace(2, 2, 2)
        y = np.linspace(-305,-301,2)
        z = np.array([(1,1),(160,160)])
        colors = np.ones((2, 2, 3), dtype=float)
        l = gl.GLSurfacePlotItem(x, y, z=z, colors=colors.reshape(2*2,3), shader='shaded', smooth=False)
        w.addItem(l)

        grid = gl.GLGridItem()
        grid.rotate(90, 0,1,0)
        grid.translate(0, 0, 105)
        grid.setSize(100,610,0)
        w.addItem(grid)

    def clear(self):
        for badminton_item in self.badminton_item_list:
            self.w.removeItem(badminton_item)
        for track_item in self.track_item_list:
            self.w.removeItem(track_item)
        for point_item in self.point_item_list:
            self.w.removeItem(point_item)
        self.badminton_item_list, self.track_item_list, self.point_item_list = [], [], []

    def clear_point(self):
        for point_item in self.point_item_list:
            self.w.removeItem(point_item)
        self.point_item_list = []

    def show_badminton_track(self, NumTrack, curve_list):
        def set_badminton_model(direction, coordinary):
            badminton_model = gl.GLMeshItem(meshdata = self.badminton_model)

            direction = direction / np.linalg.norm(direction)
            alpha1 = -np.arcsin(direction[0]) / np.pi * 180
            
            direction = direction[1:]
            if np.linalg.norm(direction) > 1e-5:
                direction = direction / np.linalg.norm(direction)
                alpha2 = np.arcsin(direction[1]) / np.pi * 180
                if direction[0] < 0:
                    alpha2 = 180 - alpha2
            else:
                alpha2 = 0   

            badminton_model.rotate(alpha1, 0, 0, 1)
            badminton_model.rotate(alpha2, 1, 0, 0)
            badminton_model.translate(*coordinary)
            self.w.addItem(badminton_model)
            self.badminton_item_list.append(badminton_model)

        self.clear()

        if NumTrack > 0 and NumTrack <= len(curve_list):  
            curve = curve_list[NumTrack - 1]
            print(curve.create_curve()[2].min())
            pts = np.vstack(curve.create_curve()).transpose()
            plt_track = gl.GLLinePlotItem(pos=pts)
            self.w.addItem(plt_track)
            self.track_item_list.append(plt_track)
            text = curve.type_list[curve.type]
            #print(text)
            #self.track_name.setText(text)
            set_badminton_model(pts[-1] - pts[-2], pts[-1])
            

    
    def show_droppoint(self, curve_list):
        colors = np.zeros((2, 2, 3), dtype=float)
        colors[...,0] = 0
        colors[...,2] = 1
        colors[...,1] = 1
        self.clear_point()
        for curve in curve_list:
            ret = curve.calc_droppoint()
            if ret is not None:
                x, y = ret
                '''
                x = int(x)
                y = int(y)
                z = np.ones((2,2))
                x1 = np.linspace(x-5, x+5, 2)
                y1 = np.linspace(y-5, y+5, 2)
                l = gl.GLSurfacePlotItem(x1, y1, z=z, colors=colors.reshape(2*2,3), shader='shaded', smooth=False)
                '''
                pos = np.empty((1, 3))
                size = np.empty((1))
                color = np.empty((1, 4))
                pos[0] = (x, y, 0)
                size[0] = 10
                color[0] = (0., 1., 0., .5)
                l = gl.GLScatterPlotItem(pos = pos, size = size, color = color, pxMode=False)
                self.point_item_list.append(l)
                self.w.addItem(l)

    def show_statistical_result(self, curve_list):
        if len(curve_list) > 0:
            count = np.zeros(len(curve_list[0].type_list))
            for curve in curve_list:
                count[curve.type] += 1

            self.plt_w.canvas.axes.clear()
            self.plt_w.canvas.axes.pie(count, labels = curve.type_list, autopct='%1.2f%%') 
            self.plt_w.canvas.draw()
            

