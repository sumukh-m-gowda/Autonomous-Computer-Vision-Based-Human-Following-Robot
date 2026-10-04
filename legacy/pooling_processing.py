from matplotlib.image import imread
import numpy as np
import os

from skimage.segmentation import slic
from skimage.util import img_as_float
import scipy

from image_processing import save_image


def watershed_pool(image_files):
    """ Pool images based on watershed segmentation. """
    for file in image_files:
        file_parts = os.path.splitext(file)
        new_filename = file_parts[0] + '_pooled' + file_parts[1]
        A = imread(file)
        A = img_as_float(A[:, :])
        segments = slic(A, n_segments=250, compactness=10, sigma=1)
        image = np.zeros((A.shape))
        for r in range(np.max(segments) + 1):
            mask = segments == r
            mask = mask.astype(int)
            mean = np.array([scipy.stats.tmean(A[:, :, 0] * mask, (0.000001, 1)),
                             scipy.stats.tmean(A[:, :, 1] * mask, (0.000001, 1)),
                             scipy.stats.tmean(A[:, :, 2] * mask, (0.000001, 1))])
            image += np.expand_dims(np.expand_dims(mean, 0), 0) * np.expand_dims(mask, 2)
        save_image(image, new_filename, as_png=True)


if __name__ == "__main__":
    test_xfiles = [
        './Data/left0586.jpg',
        './Data/left22331.jpg',
        './Data/left25343.jpg',
        './Data/left0586.jpg',
        './Data/left22331.jpg',
        './Data/left25343.jpg',
        './Data/left25057.jpg',
        './Data/left24875.jpg',
        './Data/left23945.jpg',
        './Data/left23761.jpg',
        './Data/left23075.jpg',
        './Data/left22741.jpg',
        './Data/left22088.jpg',
        './Data/left23590.jpg',
        './Data/left24496.jpg',
        './Data/left22156.jpg',
    ]
    watershed_pool(test_xfiles)
