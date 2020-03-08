# -*- coding: utf-8 -*-
"""
Copyright (C) 2019 Quentin Peter

This file is part of Worms_Inclusions.

Worms_Inclusions is distributed under CC BY-NC-SA version 4.0. You should have
recieved a copy of the licence along with Worms_Inclusions. If not, see
https://creativecommons.org/licenses/by-nc-sa/4.0/.
"""
import matplotlib.image as mpimg
from wormInclusions import (bgThreshold, Scharr_edge,
                            writeSummary, saveInclusions)
import numpy as np
from PyQt5 import QtWidgets
from glob import glob
import cv2
from matplotlib.path import Path


def getvals(bins, data):
    return np.asarray([cv2.connectedComponents(
        cv2.morphologyEx(
            cv2.morphologyEx(
                np.asarray(data > x, dtype='uint8'),
                cv2.MORPH_OPEN,
                np.ones((5, 5))),
            cv2.MORPH_CLOSE,
            np.ones((3, 3)))
    )[0] - 1 for x in bins[:-1]])


class applicationCore():

    def __init__(self, imageCanvas, inclusionCanvas, thresholdCanvas):
        self.imageCanvas = imageCanvas
        self.inclusionCanvas = inclusionCanvas
        self.thresholdCanvas = thresholdCanvas
        self.simpleThreshold = False
        self.fns = []
        self.i = 0
        self.j = 0
        self.NThreshold = True
        self.imageCanvas.createLasso(self.onLassoSelect)

    def loadImage(self, fn):
        im = mpimg.imread(fn)
        if len(im.shape) > 2:
            error = QtWidgets.QMessageBox()
            error.critical(self.imageCanvas,
                           "Error",
                           "An invalid image is present."
                           " Please don't use images stacks or RGB images.")
            error.setFixedSize(500, 200)
            return
        self.fn = fn
        self.setimage(im)
        self.originalim = im

    def setimage(self, im):
        self.im = im
        self.setWormThreshold(bgThreshold(self.im))
        self.grad = Scharr_edge(self.im)[0]
        self.updateThreshold()
        self.imageCanvas.setimage(self.im)

    def updateThreshold(self, threshold=None):
        if self.simpleThreshold:
            data = self.im
        else:
            data = self.grad

        if self.NThreshold:
            n = 10
            bins = np.arange(n + 1) / n * (np.nanmax(data) -
                                           np.nanmin(data)) + np.nanmin(data)
            vals = getvals(bins, data)
            mymin = np.min(bins[:-1][vals > 0])
            mymax = np.max(bins[1:][vals > 0])
            n = 100
            bins = np.arange(n + 1) / n * (mymax - mymin) + mymin
            vals = getvals(bins, data)

            maxarg = np.argwhere(vals > 0)[-1][0]
            self.thresholdCanvas.setData(vals[:maxarg], bins[:maxarg + 1])
            if threshold is None:
                threshold = bins[(maxarg + 1) // 2]
        else:
            self.thresholdCanvas.setData(*np.histogram(data, 100))

        if threshold is None:
            threshold = np.max(data) / 2
        self.setThreshold(threshold)

    def setThreshold(self, threshold):
        self.threshold = threshold
        self.thresholdCanvas.setThreshold(threshold)
        if self.simpleThreshold:
            maskIncl = self.im > threshold
        else:
            maskIncl = self.grad > threshold
        maskIncl = np.asarray(maskIncl, dtype='uint8')

        maskIncl = cv2.morphologyEx(maskIncl, cv2.MORPH_OPEN, np.ones((5, 5)))
        maskIncl = cv2.morphologyEx(maskIncl, cv2.MORPH_CLOSE, np.ones((3, 3)))

        self.labels = cv2.connectedComponents(maskIncl)[1]
        self.inclusionCanvas.setLabels(self.labels)

    def setWormThreshold(self, threshold):
        self.bgThreshold = threshold
        self.thresholdCanvas.setThreshold(threshold)
        self.maskWorm = self.im > threshold
        self.inclusionCanvas.setMaskWorm(self.maskWorm)

    def onThresClick(self, event):
        X = event.xdata
        if X is None:
            return None
        if self.iseditingbg():
            self.setWormThreshold(event.xdata)
        else:
            self.setThreshold(event.xdata)

    def onThreshType(self, checked):
        self.simpleThreshold = checked
        self.updateThreshold()

    def onThreshDisplay(self, checked):
        self.NThreshold = checked
        self.updateThreshold(self.threshold)

    def onImageDisplay(self, checked):
        if checked:
            self.imageCanvas.setimage(self.grad)
        else:
            self.imageCanvas.setimage(self.im)

    def onEditWorm(self, checked):
        if checked:
            self.thresholdCanvas.setData(*np.histogram(self.im, 100))
            self.thresholdCanvas.setThreshold(self.bgThreshold)
        else:
            self.updateThreshold(self.threshold)

    def onOpenFile(self):
        fn = QtWidgets.QFileDialog.getExistingDirectory()
        if fn is not None and fn != '':
            self.loadFolder(fn)

    def normalizeIm(self):
        self.im = self.im / self.im[~self.maskWorm].mean()

    def getum2px2ratio(self):
        return float(self.getRatioText())

    def loadFolder(self, folder):
        if folder[-1] != '/':
            folder += '/'
        self.fns = sorted(glob(folder + '*.tif'))
        print(folder, str(len(self.fns)))
        if len(self.fns) > 0:
            self.i = 0
            self.j = 0
            self.loadImage(self.fns[self.i])
            self.im_npwp = []
            self.im_intensity = []
            self.im_size = []
            self.im_index = []
            self.skipped = []

    def onNext(self):
        self.save()
        self.loadNext()

    def onSame(self):
        self.save()
        self.loadSame()

    def save(self):
        imagesFolder = self.fn[:self.fn.rfind('/')] + '/images_script/'
        saveInclusions(self.im, self.labels, self.maskWorm, imagesFolder,
                       self.fn, self.i, self.getum2px2ratio(),
                       self.im_npwp, self.im_intensity, self.im_size,
                       self.im_index, self.j)

    def onImageClick(self, event):
        if event.xdata is None or event.ydata is None:
            return None
        self.X0roi = (int(event.xdata), int(event.ydata))

    def onImageMove(self, event):
        if event.xdata is None or event.ydata is None:
            return None
        if hasattr(self, 'X0roi'):
            self.X1roi = (int(event.xdata), int(event.ydata))
            X = (np.min((self.X0roi[0], int(event.xdata))),
                 np.min((self.X0roi[1], int(event.ydata))))
            W = np.abs(self.X0roi[0] - int(event.xdata))
            H = np.abs(self.X0roi[1] - int(event.ydata))

            self.imageCanvas.addRectangle(X, W, H)

    def onImageRelease(self, event):
        xd = event.xdata
        yd = event.ydata
        if xd is None or yd is None:
            xd, yd = self.X1roi
        X = sorted((self.X0roi[0], int(xd)))
        Y = sorted((self.X0roi[1], int(yd)))
        if Y[1] - Y[0] > 1 and X[1] - X[0] > 1:
            self.setimage(self.im[Y[0]:Y[1], X[0]:X[1]])
        else:
            self.imageCanvas.standalone()
        del self.X0roi

    def onLassoSelect(self, poly_verts):
        ny, nx = self.im.shape
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()

        points = np.vstack((x, y)).T

        mask = Path(poly_verts).contains_points(points)
        mask = mask.reshape((ny, nx))

        im = np.asarray(mpimg.imread(self.fns[self.i]), float)
        im[np.logical_not(mask)] = np.nan
        self.setimage(im)

    def onROIReset(self):
        self.setimage(self.originalim)

    def onSkip(self):
        self.skipped.append(self.i)
        self.loadNext()

    def loadNext(self):
        self.i += 1
        self.j = 0
        if self.i < len(self.fns):
            self.loadImage(self.fns[self.i])
        else:
            self.onEnd()

    def loadSame(self):
        self.j += 1
        self.loadImage(self.fns[self.i])

    def onEnd(self):
        writeSummary(self.fn[:self.fn.rfind('/')] + '/summary.txt',
                     self.im_npwp, self.im_intensity, self.im_size,
                     self.im_index,
                     self.fns, self.skipped)
        for i in range(self.i, len(self.fns)):
            self.skipped.append(i)
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText('Done!')
        msgBox.exec_()
