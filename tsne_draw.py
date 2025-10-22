import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import sys
import os
import pickle
from scipy.spatial import Delaunay
import numpy as np


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, bg_color, dot_color, line_color, tsne, tsneRect, tri, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.subplots_adjust(
            left=0, right=1, top=1, bottom=0, wspace=0, hspace=0
        )
        self.axes.set_axis_off()
        self.axes.triplot(tsne[:,0], tsne[:,1], tri.simplices, color=line_color)
        self.axes.scatter(tsne[:,0], tsne[:,1], color=dot_color)
        fig.set_facecolor(bg_color)
        super().__init__(fig)

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        filePath = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(filePath, 'data', 'thumbs-clean', 'delaunay.pkl'), 'rb') as f:
            tsne, tsneRect, tri = pickle.load(f)
        bg = [46/255] * 3 # Monokai grey
        line = [214/255] * 3 # Monokai
        sc = MplCanvas(bg, line, line + [0.2], tsne, tsneRect, tri, self, width=5, height=4, dpi=50)
        self.setCentralWidget(sc)

        self.show()

def plot_tsne(width, height, bg_color, dot_color, line_color, tsne, tsneRect, tri):
    dpi = 200
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.exec_()