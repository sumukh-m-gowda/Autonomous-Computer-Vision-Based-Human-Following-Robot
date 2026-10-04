from matplotlib.image import imread
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import os
import numpy as np
import softmax_reg


def CompressImage(A, K, tol=1e-3, max_iter=500):
    """
    Compress image into compact form (K-means on pixel colours).

    :param A: original image
    :param K: # of clusters to use
    :param tol: relative tolerance for convergence
    :param max_iter: maximum number of iterations
    :return: compressed image
    """
    A, w1, w2, n1, n2 = flatten(A)

    # Initialize mean values randomly
    mu = np.random.randint(0, high=256, size=(K, n2))
    w = np.zeros(n1)
    w_ij = np.zeros((n1, K))

    it = 0
    error = np.inf
    while (it < max_iter) and (error > tol):
        prev_mu = mu

        # Assign each pixel to its nearest mean
        for ii in range(n1):
            w[ii] = np.argmin(np.sum(np.square(A[ii, :] - mu), axis=1))

        # One-hot assignment matrix
        for jj in range(K):
            w_ij[:, jj] = 1 * (w == jj)

        # Recompute means
        mu_1 = w_ij.T @ A
        mu_2 = np.sum(w_ij, axis=0)
        mu = (mu_1.T / mu_2.T).T

        it += 1
        error = np.linalg.norm(mu - prev_mu) / np.linalg.norm(mu)

    A_comp = mu[w.astype(int)].astype(int)
    A_comp = A_comp.reshape((w1, w2, n2))
    return A_comp


def resize_img(filename, width, height):
    """Resize an image, save it next to the original as *_red.*, and reload it."""
    file_parts = os.path.splitext(filename)
    new_filename = file_parts[0] + '_red' + file_parts[1]
    img = Image.open(filename)
    img.resize((width, height), PIL.Image.LANCZOS).save(new_filename)
    return imread(new_filename)


def ConvoluteImage(A, R):
    """Give every pixel the colours of its RxR neighbourhood as extra features.

    Output shape is (w1, w2, R*R*channels), then edges are trimmed because
    np.roll wraps around the border and contaminates those pixels.
    """
    R_half = int((R - 1) / 2)
    A_flat, w1, w2, n1, n2 = flatten(A)
    n2_new = R ** 2 * n2
    A_conv = -np.ones((w1, w2, n2_new))
    inc = 0
    for ii in range(-R_half, R_half + 1):
        for jj in range(-R_half, R_half + 1):
            A_conv[:, :, range(inc, inc + n2)] = np.roll(A, shift=(ii, jj), axis=(0, 1))
            inc = inc + n2
    return trim_edges(A_conv, R)


def trim_edges(A, R):
    R_half = int((R + 1) / 2)
    w1 = A.shape[0]
    w2 = A.shape[1]
    A = A[R_half:(w1 - R_half), R_half:(w2 - R_half), :]
    return A


def DeconvoluteImage(A_conv, R, n_val):
    center = int(np.round(R ** 2 / 2))
    return A_conv[:, :, range(center, center + n_val)]


def ConvertWithKey(A, key):
    """ Replace pixel values with key index """
    A, w1, w2, n1, n2 = flatten(A)
    A = np.expand_dims(A, 1)
    key = np.expand_dims(key, 0)
    dists = np.sum(np.square(A - key), axis=2)
    A_new = np.argmin(dists, axis=1)
    A_new = unflatten(A_new, w1, w2)
    return A_new


def ConvertWithKeyFlat(A, key):
    """ Replace each (flattened) pixel with the index of the nearest key colour """
    A = np.expand_dims(A, 1)
    key = np.expand_dims(key, 0)
    dists = np.sum(np.square(A - key), axis=2)
    A_new = np.argmin(dists, axis=1)
    return A_new


def ReplaceWithKey(A, key):
    """ Replace key indices with pixel values """
    return key[A, :]


def flatten(A):
    """ Change from rectangular image to vector of pixels """
    w1 = A.shape[0]
    w2 = A.shape[1]
    n1 = w1 * w2
    n2 = 0 if len(A.shape) <= 2 else A.shape[2]
    A = np.array(A.reshape(n1)) if not n2 else np.array(A.reshape((n1, n2)))
    return A, w1, w2, n1, n2


def unflatten(A, w1, w2):
    """ Change from vector of pixels to rectangular image """
    if len(A.shape) > 1:
        return np.array(A.reshape((w1, w2, A.shape[1])))
    else:
        return np.array(A.reshape((w1, w2)))


def save_image(A, name, as_png=False):
    """Save image to output folder"""
    if as_png:
        file_parts = os.path.splitext(name)
        name = file_parts[0] + '.png'
    fig = plt.imshow(A)
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.close()


def train_classifier(train_xfiles, train_yfiles, basewidth, baseheight, mu_stored, R):
    """ Train softmax classifier on list of files """
    # Build the big training matrix: one row per pixel, R*R*3 features
    for ii, file in enumerate(train_xfiles):
        if ii == 0:
            A = resize_img(file, basewidth, baseheight)
            A = ConvoluteImage(A, R)
            A, w1, w2, n1, n2 = flatten(A)
            A_train = np.zeros((len(train_xfiles) * n1, n2))
            A_train[0:n1, :] = A
        else:
            A = resize_img(file, basewidth, baseheight)
            A = ConvoluteImage(A, R)
            A, _, _, _, _ = flatten(A)
            A_train[(ii * n1):((ii + 1) * n1), :] = A

    # Build the label matrix from the hand-painted masks (same trimming as above)
    for ii, file in enumerate(train_yfiles):
        if ii == 0:
            A = resize_img(file, basewidth, baseheight)
            A = trim_edges(A, R)
            A, w1, w2, n1, n2 = flatten(A)
            A_sup = np.zeros((len(train_yfiles) * n1, n2))
            A_sup[0:n1, :] = A
        else:
            A = resize_img(file, basewidth, baseheight)
            A = trim_edges(A, R)
            A, _, _, _, _ = flatten(A)
            A_sup[(ii * n1):((ii + 1) * n1), :] = A

    # Snap each mask pixel to the nearest class colour -> integer class label
    A_sup = ConvertWithKeyFlat(A_sup, mu_stored)
    return softmax_reg.train_softmax(A_train, A_sup)


def predict_multiple(mods, file_list, mu_stored, R, basewidth, baseheight):
    os.makedirs('./Output', exist_ok=True)
    for ii, file in enumerate(file_list):
        print(f"Predicting {ii + 1}/{len(file_list)}: {file}", flush=True)
        A = resize_img(file, basewidth, baseheight)
        A = ConvoluteImage(A, R)
        A, w1, w2, n1, n2 = flatten(A)
        A_pred = softmax_reg.eval_softmax(mods, A)
        A_pred = ReplaceWithKey(A_pred, mu_stored)
        A_pred = unflatten(A_pred, w1, w2)
        save_image(A_pred, './Output/' + os.path.basename(file))


if __name__ == "__main__":
    basewidth = 200
    baseheight = 112

    train_xfiles = [
        './Data/left0586.jpg',
        './Data/left25343.jpg',
        './Data/left22331.jpg',
        './Data/left25057.jpg',
        './Data/left24875.jpg',
        './Data/left23945.jpg',
        './Data/left23761.jpg',
        './Data/left23075.jpg',
        './Data/left22741.jpg',
        './Data/left22088.jpg',
        './Data/left24496.jpg',
    ]
    test_xfiles = [
        './Data/left23590.jpg',
        './Data/left22156.jpg',
    ]
    train_yfiles = [
        './Data/left0586_sup.jpg',
        './Data/left25343_sup.jpg',
        './Data/left22331_sup.jpg',
        './Data/left25057_sup.jpg',
        './Data/left24875_sup.jpg',
        './Data/left23945_sup.jpg',
        './Data/left23761_sup.jpg',
        './Data/left23075_sup.jpg',
        './Data/left22741_sup.jpg',
        './Data/left22088_sup.jpg',
        './Data/left24496_sup.jpg',
    ]

    # Class colours (RGB): black=sky, red=traversable, green=putting green,
    # purple=untraversable, yellow=pavement, blue=golfer
    mu_stored = np.array([[0, 0, 0], [237, 27, 36], [35, 177, 77],
                          [163, 73, 163], [253, 240, 11], [63, 71, 204]])
    R = 17  # convolution window side length (17x17 pixels -> 867 features)

    print("Training...", flush=True)
    mods = train_classifier(train_xfiles, train_yfiles, basewidth, baseheight, mu_stored, R)
    print("Training done. Predicting...", flush=True)
    predict_multiple(mods, train_xfiles + test_xfiles, mu_stored, R, basewidth, baseheight)
    print("Done. Results are in ./Output", flush=True)