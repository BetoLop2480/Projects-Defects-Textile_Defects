import numpy as np
import os

import cv2 as cv




def img_write(path2Save, img, adjust=False):
    """
    To write an image using OpenCV
    :param path2Save:
    :param img:
    :return:
    """

    #Writing an image on disk
    img_Copy = np.zeros_like(img)
    if adjust:
         cv.normalize(img, img_Copy, 0, 255, cv.NORM_MINMAX,
                            dtype=cv.CV_32F) # Adjusting as simple image just for visualization
    else:
        img_Copy = np.copy(img)

    img_Copy = np.uint8(img_Copy)
    cv.imwrite(path2Save, img_Copy)

    return None


def img_read(path2Read, colorSpace= cv.IMREAD_COLOR):
    """
    To read an image using OpenCV
    :param path2Read:
    :param colorSpace:
    :return:
    """
    #Reading an image)
    img = cv.imread(path2Read, colorSpace)

    if img is None:
        raise Exception("No image read...")

    return img


def lib_read_from_folder(path):
    """
    Function to retrive the names and extensions of all in a folder
    :param path:
    :return:
    """

    list_of_files = os.listdir(path)

    return list_of_files


def lib_create_folder(path):
    """
    Function to create just one folder at a time
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    return None