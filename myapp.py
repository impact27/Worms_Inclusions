# -*- coding: utf-8 -*-
"""
Copyright (C) 2019 Quentin Peter

This file is part of Worms_Inclusions.

Worms_Inclusions is distributed under CC BY-NC-SA version 4.0. You should have
recieved a copy of the licence along with Worms_Inclusions. If not, see
https://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import matplotlib
matplotlib.use('Qt5Agg')


import os
import sys
from PyQt5 import QtCore, QtWidgets, QtGui
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from applicationCore import applicationCore
from matplotlib.patches import Rectangle
from matplotlib.widgets import LassoSelector



progname = os.path.basename(sys.argv[0])
progversion = "0.1"


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class imageCanvas(MyMplCanvas):
    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        self.axes = self.figure.add_subplot(111)
        self.cbar = None


    def createLasso(self, onLassoSelect):
        self.lasso = LassoSelector(self.axes, onLassoSelect,
                                   lineprops=dict(color='w'))

    def setimage(self, im):
        self.im = im
#        if hasattr(self, 'ROIrect'):
#            del self.ROIrect
        self.update_figure()

    def update_figure(self):
        self.axes.clear()
        mp = self.axes.imshow(self.im)
        self.axes.axis('image')
        if self.cbar is not None:
            self.cbar.mappable = mp
        else:
            self.cbar = self.figure.colorbar(mp)
#        if hasattr(self, 'ROIrect'):
#
#            self.axes.add_patch(self.ROIrect)
        self.draw()

    def standalone(self):
        if hasattr(self, 'im'):
            plt.figure()
            plt.imshow(self.im)
            plt.colorbar()
            plt.show()

    def addRectangle(self, X, W, H):
#        self.ROIrect = Rectangle(X, W, H, facecolor='none', edgecolor='white')
        self.update_figure()


class inclusionCanvas(MyMplCanvas):
    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)

    def setLabels(self, labels):
        self.labels = np.array(labels, dtype=float)
        self.labels[labels == 0] = np.nan
        self.update_figure()

    def setMaskWorm(self, maskWorm):
        self.maskWorm = maskWorm
        self.update_figure()

    def update_figure(self, axes=None):
        if axes is None:
            axes = self.axes
        axes.cla()
        if hasattr(self, 'labels'):
            axes.imshow(self.labels)
            n = np.nanmax(self.labels)
            if np.isnan(n):
                n = 0
            axes.set_title(str(int(n)) + ' Inclusions')
        if hasattr(self, 'maskWorm'):
            axes.imshow(self.maskWorm, alpha=.5, cmap=plt.get_cmap('Reds'))
        axes.axis('image')
        self.draw()

    def standalone(self, event):
        fig = plt.figure()
        axes = fig.add_subplot(111)
        self.update_figure(axes)
        plt.show()


class thresholdCanvas(MyMplCanvas):
    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        self.log = False

    def setData(self, data, bins):
        self.data = data
        self.bins = bins
        self.threshold = 0

    def update_figure(self):
        self.axes.cla()
        self.axes.bar(self.bins[:-1], self.data, self.bins[1] - self.bins[0])
        self.axes.plot([self.threshold, self.threshold],
                       [0, self.axes.axis()[-1]], 'g')
        self.axes.set_yscale("log")
        if self.log:
            self.axes.set_xscale("log")
        self.axes.set_xlabel('Value')
        self.draw()

    def setThreshold(self, threshold):
        self.threshold = threshold
        if hasattr(self, 'data'):
            self.update_figure()

    def onXLog(self, checked):
        self.log = checked
        self.update_figure()


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):

        # Init everything
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        # create menu
        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

        self.statusBar().showMessage("Hello", 2000)

        # Create canevas
        self.imageCanvas = imageCanvas()
        self.thresholdCanvas = thresholdCanvas()
        self.inclusionCanvas = inclusionCanvas()

        # create appication core
        self.applicationCore = applicationCore(self.imageCanvas,
                                               self.inclusionCanvas,
                                               self.thresholdCanvas)

        # Create Buttons
        logbutton = QtWidgets.QPushButton('Threshold xlog')
        logbutton.setCheckable(True)

        thresTypebutton = QtWidgets.QPushButton('Intensity threshold')
        thresTypebutton.setCheckable(True)

        gradbutton = QtWidgets.QPushButton('Show gradient')
        gradbutton.setCheckable(True)

        bgbutton = QtWidgets.QPushButton('Edit worm')
        bgbutton.setCheckable(True)

        openButton = QtWidgets.QPushButton('Load Folder')
        nextButton = QtWidgets.QPushButton('Save/Next')
        sameButton = QtWidgets.QPushButton('Save/Same')
        skipButton = QtWidgets.QPushButton('Skip')
        endButton = QtWidgets.QPushButton('End')

        ResetROIButton = QtWidgets.QPushButton('Reset ROI')

        scaleLine = QtWidgets.QLineEdit()
        scaleLine.setValidator(QtGui.QDoubleValidator(0, 100, 3))
        scaleLine.setMaximumWidth(100)
        scaleLine.setText('1')
        # add method to applicationCore
        self.applicationCore.iseditingbg = bgbutton.isChecked
        self.applicationCore.getRatioText = scaleLine.text

        # Create main widget
        self.main_widget = QtWidgets.QWidget(self)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        # Connections
        self.thresholdCanvas.mpl_connect('button_press_event',
                                         self.applicationCore.onThresClick)
#        self.imageCanvas.mpl_connect('button_press_event',
#                                     self.applicationCore.onImageClick)
#        self.imageCanvas.mpl_connect('motion_notify_event',
#                                     self.applicationCore.onImageMove)
#        self.imageCanvas.mpl_connect('button_release_event',
#                                     self.applicationCore.onImageRelease)
        self.inclusionCanvas.mpl_connect('button_press_event',
                                         self.inclusionCanvas.standalone)
        bgbutton.clicked.connect(self.applicationCore.onEditWorm)
        gradbutton.clicked.connect(self.applicationCore.onImageDisplay)
        thresTypebutton.clicked.connect(self.applicationCore.onThreshType)
        logbutton.clicked.connect(self.thresholdCanvas.onXLog)
        openButton.clicked.connect(self.applicationCore.onOpenFile)
        nextButton.clicked.connect(self.applicationCore.onNext)
        sameButton.clicked.connect(self.applicationCore.onSame)
        ResetROIButton.clicked.connect(self.applicationCore.onROIReset)
        skipButton.clicked.connect(self.applicationCore.onSkip)
        endButton.clicked.connect(self.applicationCore.onEnd)

        # Layout
        vert = QtWidgets.QVBoxLayout(self.main_widget)
        h0 = QtWidgets.QHBoxLayout()
        h1 = QtWidgets.QHBoxLayout()

        h0.addWidget(self.imageCanvas)
        h0.addWidget(self.inclusionCanvas)
        h1.addWidget(self.thresholdCanvas)

        vButton = QtWidgets.QVBoxLayout()
        vButton.addWidget(thresTypebutton)
        vButton.addWidget(logbutton)
        vButton.addWidget(gradbutton)
        vButton.addWidget(bgbutton)
        vButton.addWidget(openButton)
        vButton.addWidget(nextButton)
        vButton.addWidget(sameButton)
        vButton.addWidget(ResetROIButton)
        vButton.addWidget(scaleLine)
        vButton.addWidget(skipButton)
        vButton.addWidget(endButton)

        h1.addLayout(vButton)

        vert.addLayout(h0)
        vert.addLayout(h1)

        self.activateWindow()
        self.setFocus()

        self.applicationCore.onOpenFile()

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """hello""")


qApp = QtWidgets.QApplication(sys.argv)

aw = ApplicationWindow()
aw.setWindowTitle("%s" % progname)
aw.show()
sys.exit(qApp.exec_())
