# -*- coding: utf-8 -*-
"""
Copyright (C) 2019 Quentin Peter

This file is part of Worms_Inclusions.

Worms_Inclusions is distributed under CC BY-NC-SA version 4.0. You should have
recieved a copy of the licence along with Worms_Inclusions. If not, see
https://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

# Configuration (use * for any matching patern)


um2px2ratio = 1
folders = 'abeta/'
files = '*.tif'
threshold = 20
filtR = 1
simpleThreshold = False
normalizeWithBg = False

# fillHoles=True
# simpleBG=True

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Import everything needed
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, imshow, plot, close, show, colorbar
import matplotlib.image as mpimg
import numpy as np
from glob import glob
import cv2
import scipy.ndimage.morphology as mpy
import scipy.ndimage.measurements as msr
from scipy.stats import sem
import scipy
from numpy.fft import fft2, fftshift
gfilter = scipy.ndimage.filters.gaussian_filter1d
from scipy.signal import correlate2d
from skimage.morphology import convex_hull_image
from skimage.feature import blob_log
from skimage.feature import peak_local_max
import os


def Scharr_edge(im):
    im = cv2.GaussianBlur(np.asarray(im, dtype="float32"), (filtR, filtR), 0)
    Gx = cv2.Scharr(im, -1, 0, 1)
    Gy = cv2.Scharr(im, -1, 1, 0)
    res = cv2.magnitude(Gx, Gy)
    angle = np.arctan2(Gy, Gx)
    return res, angle


def bgThreshold(im):
    bgIntensity = np.bincount(
        np.ravel(np.array(im[np.isfinite(im)], dtype=int)))[:-1].argmax()
    return 2 * bgIntensity - np.nanmin(im)


def scharrMethod(im, threshold=5):
    """Compute the norm of the gradient of the image and apply threshold
    """

    # Get gradient
    res, __ = Scharr_edge(im)

    # Apply threshold
    b = res > threshold
    b = mpy.binary_opening(b, np.ones((5, 5)))
    b = mpy.binary_closing(b, np.ones((3, 3)))

    return b


def writeFileInfos(fn, intensity_inclusions, size_inclusions, prct):
    nl = len(intensity_inclusions)
    with open(fn, 'w') as f:
        f.write('Worm detected coverage: %.1f\n' % (prct * 100))
        f.write('Number of inclusions: %d\n' % nl)
        if nl > 0:
            f.write('Normalized Intensity Mean: %f SD: %f SEM: %f\n' % (
                    intensity_inclusions.mean(),
                    intensity_inclusions.std(),
                    sem(intensity_inclusions)))
            f.write('Size [px^2] Mean: %f SD: %f SEM: %f\n' % (
                    size_inclusions.mean(),
                    size_inclusions.std(),
                    sem(size_inclusions)))
        f.write('N\tIntensity\tSize\n')
        for i in range(nl):
            f.write(
                '%d\t%f\t%f\n' %
                (i, intensity_inclusions[i], size_inclusions[i]))


def writeSummary(fn, im_number_per_worm_pixel, im_intensity, im_size, im_index,
                 fns, skipped=[]):
    # assert(len(fns)==len(skipped)+len(im_number_per_worm_pixel))
    with open(fn, 'w') as f:
        f.write('Number of inclusions per worm pixels Mean:' +
                ' %f SD: %f SEM: %f\n' % (
                    np.mean(im_number_per_worm_pixel),
                    np.std(im_number_per_worm_pixel),
                    sem(im_number_per_worm_pixel)))
        f.write('Inclusions intensity Mean: %f SD: %f SEM: %f\n' % (
            np.mean(im_intensity),
            np.std(im_intensity),
            sem(im_intensity)))
        f.write('Inclusions size Mean: %f SD: %f SEM: %f\n\n' % (
            np.mean(im_size),
            np.std(im_size),
            sem(im_size)))
        for i in range(len(fns)):
            if i not in skipped:
                f.write('%d: %s\n' % (i, fns[i]))
        f.write('file\tnumber\tNPWP\tInt\tSize\n')
        for idx, num, intens, size in zip(im_index, im_number_per_worm_pixel,
                                          im_intensity, im_size):
            f.write(
                    '%d\t%d\t%f\t%f\t%f\n' %
                    (*idx, num, intens, size))


def saveInclusions(im, labels, maskWorm, imagesFolder, fn, i, um2px2ratio=1,
                   im_number_per_worm_pixel=[], im_intensity=[], im_size=[],
                   im_index=[], j=0):
    if not os.path.isdir(imagesFolder):
        os.mkdir(imagesFolder)

    imagesFolder = imagesFolder + str(i)
    if j > 0:
        imagesFolder = imagesFolder + '_{}'.format(j)
    # get worm coverage
    prct = maskWorm.sum() / np.product(im.shape)

    nl = labels.max()

    # extract normalized intensity
    intensity_inclusions = msr.mean(im, labels, np.arange(nl) + 1)
    # extract size
    size_inclusions = np.bincount(labels.flat)[1:] * um2px2ratio

    # plot image
    f = figure()
    imshow(im)
    colorbar()
    # if no inclusions
    if nl < 1:
        plt.title(fn + ', no Inclusions, %.1f %% worm' % (prct * 100))
        plt.savefig(imagesFolder + '_image')
        f.close()
    else:

        # Fill array
        im_number_per_worm_pixel.append(nl / maskWorm.sum())
        im_intensity.append(intensity_inclusions.mean())
        im_size.append(size_inclusions.mean())
        im_index.append((i, j))

        plt.title(fn + ', %.1f %% worm' % (prct * 100))
        plt.savefig(imagesFolder + '_image')
        plt.close()

        # plot detected inclusions
        # coordinates = peak_local_max(im, min_distance=20)
        f = figure()
        imshow(labels)
        imshow(maskWorm, alpha=.5)
        # plot(coordinates[:, 1], coordinates[:, 0], 'w.')
        plt.title(
            "%d inclusions, mean = %.2f" %
            (nl, intensity_inclusions.mean()))
        plt.savefig(imagesFolder + '_inclusions')
        plt.close()

        # plot intensity
        f = figure()
        plt.hist(intensity_inclusions, 20)
        plt.xlabel('Normalized intensity')
        plt.ylabel('Number of inclusions')
        plt.savefig(imagesFolder + '_intensityHistogram')
        plt.close()

        f = figure()
        plt.hist(size_inclusions, 20)
        plt.xlabel('Size')
        plt.ylabel('Number of inclusions')
        plt.savefig(imagesFolder + '_sizeHistogram')
        plt.close()

    writeFileInfos(os.path.splitext(fn)[0] + "_{}.txt".format(j),
                   intensity_inclusions, size_inclusions, prct)


if __name__ == '__main__':
    for folder in glob(folders):
        if folder[-1] != '/':
            folder += '/'
        imagesFolder = folder + 'images_script/'
        # Load images
        fns = sorted(glob(folder + files))

        im_number_per_worm_pixel = []
        im_intensity = []
        im_size = []

        for i, fn in enumerate(fns):
            im = mpimg.imread(fn)
            # plot image
            maskWorm = im > bgThreshold(im)

            # get worm coverage
            prct = maskWorm.sum() / np.product(im.shape)

            if normalizeWithBg:
                # Normalize the image with background mean intensity
                im = im / im[~maskWorm].mean()

            if simpleThreshold:
                maskIncl = im > threshold
            else:
                maskIncl = scharrMethod(im, im[~maskWorm].mean() * threshold)

            # Get the labels
            labels, nl = msr.label(maskIncl)

            saveInclusions(
                im, labels, maskWorm, imagesFolder, fn, i, um2px2ratio,
                im_number_per_worm_pixel, im_intensity, im_size)

        writeSummary(
            folder +
            'summary.txt',
            im_number_per_worm_pixel,
            im_intensity,
            im_size,
            fns)
