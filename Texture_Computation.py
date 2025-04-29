import cv2
import numpy as np


import cv2 as cv

import Basics




def co_ocurrence_featureVector(mtx):
    """
    Constructing feature vector by image.
    :param mtx:
    :return:
    """

    energia = co_ocurrence_energia(mtx)
    contraste = co_ocurrence_contraste(mtx)
    homog = co_ocurrence_homogeneidad(mtx)
    entropia = co_ocurrence_entropia(mtx)
    disimil = co_ocurrence_disimilaridad(mtx)

    return energia, contraste, homog, entropia, disimil

def co_ocurrence_energia(mtx):
    """
    Calcular energia de una matriz de Co-ocurrencia.
    :param mtx:
    :return: Valor de energia
    """
    energia = np.sum(mtx ** 2)

    return energia

def co_ocurrence_contraste(mtx):
    """
    Calcular contraste de una matriz de Co-ocurrencia.
    :param mtx:
    :return: Valor de energia
    """
    N = mtx.shape[0]
    contraste = 0.0

    for i in range(N):
        for j in range(N):
            contraste += mtx[i, j] * (i - j) ** 2

    return contraste

def co_ocurrence_homogeneidad(mtx):
    """
    Calcular homogeneidad de una matriz de Co-ocurrencia.
    :param mtx:
    :return: Valor de energia
    """
    N = mtx.shape[0]
    homogeneidad = 0.0

    for i in range(N):
        for j in range(N):
            homogeneidad += mtx[i, j] / (1 + np.abs(i -j))

    return homogeneidad

def co_ocurrence_entropia(mtx):
    """
    Calcular entropia de una matriz de Co-ocurrencia.
    :param mtx:
    :return: Valor de energia
    """
    entropia = 0.0
    entropia = np.sum(mtx * np.log(mtx + 1e-06))

    return entropia

def co_ocurrence_disimilaridad(mtx):
    """
    Calcular disimil de una matriz de Co-ocurrencia.
    :param mtx:
    :return: Valor de energia
    """
    N = mtx.shape[0]
    disimil = 0.0

    for i in range(N):
        for j in range(N):
            disimil += mtx[i, j] * np.abs(i - j)

    return disimil



def co_ocurrence_matrix0(img, distance=1):
    """
    The quantization of the gray level is 0-255.
    :param img: Image to analyze
    :param distance: In pixels
    :param theta: Orientation
    :return: Matrix of co-ocurrence
    """

    mtx_coOcurrence = np.zeros((256, 256), dtype=np.float32)

    # Retrieving dimension of the image to analyze
    h = img.shape[0]
    w = img.shape[1]

    for i in range(distance, h-distance):
        for j in range(distance, w-distance):
            # 0-degrees
            mtx_coOcurrence[int(img[i, j]), int(img[i, j+distance])] += 1

    mtx_coOcurrence /= np.sum(mtx_coOcurrence)

    return mtx_coOcurrence

def co_ocurrence_matrix45(img, distance=1):
    """
    The quantization of the gray level is 0-255.
    :param img: Image to analyze
    :param distance: In pixels
    :param theta: Orientation
    :return: Matrix of co-ocurrence
    """

    mtx_coOcurrence = np.zeros((256, 256), dtype=np.float32)

    # Retrieving dimension of the image to analyze
    h = img.shape[0]
    w = img.shape[1]

    for i in range(distance, h-distance):
        for j in range(distance, w-distance):
            # 45-degrees
            mtx_coOcurrence[int(img[i, j]), int(img[i-distance, j+distance])] += 1

    mtx_coOcurrence /= np.sum(mtx_coOcurrence)

    return mtx_coOcurrence


def co_ocurrence_matrix90(img, distance=1):
    """
    The quantization of the gray level is 0-255.
    :param img: Image to analyze
    :param distance: In pixels
    :param theta: Orientation
    :return: Matrix of co-ocurrence
    """

    mtx_coOcurrence = np.zeros((256, 256), dtype=np.float32)

    # Retrieving dimension of the image to analyze
    h = img.shape[0]
    w = img.shape[1]

    for i in range(distance, h-distance):
        for j in range(distance, w-distance):
            # 90-degrees
            mtx_coOcurrence[int(img[i, j]), int(img[i-distance, j])] += 1

    mtx_coOcurrence /= np.sum(mtx_coOcurrence)

    return mtx_coOcurrence


def co_ocurrence_matrix135(img, distance=1):
    """
    The quantization of the gray level is 0-255.
    :param img: Image to analyze
    :param distance: In pixels
    :param theta: Orientation
    :return: Matrix of co-ocurrence
    """

    mtx_coOcurrence = np.zeros((256, 256), dtype=np.float32)

    # Retrieving dimension of the image to analyze
    h = img.shape[0]
    w = img.shape[1]

    for i in range(distance, h-distance):
        for j in range(distance, w-distance):
            # 135-degrees
            mtx_coOcurrence[int(img[i, j]), int(img[i-distance, j-distance])] += 1

    mtx_coOcurrence /= np.sum(mtx_coOcurrence)

    return mtx_coOcurrence


