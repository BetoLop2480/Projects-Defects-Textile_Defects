import numpy as np



import cv2 as cv

import Basics
import Texture_Computation as text

def test_function():

    img_path = 'C:/Users/alber/Documents/Textile_defects/Project_Data/TILDA_400/hole/014_patch1-5.png'

    print(img_path)
    img = Basics.img_read(img_path, cv.IMREAD_GRAYSCALE)


    mtx = text.co_ocurrence_matrix0(img, distance=1)


    mtx2 = np.zeros_like(mtx)
    cv.normalize(mtx, mtx2, 0, 255, cv.NORM_MINMAX, cv.CV_32F)

    mtx2 = mtx2.astype(np.uint8)
    cv.imshow("Image", img)
    cv.imshow("Matrix", mtx2)
    cv.waitKey()


    return None