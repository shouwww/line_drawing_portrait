import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
import cv2
import datetime
import numpy as np
import math


class Img_Process_Tool:
    _face_cascade_path = '../libs/haarcascades/haarcascade_frontalface_default.xml'
    _eye_cascade_path = '../libs/haarcascades/haarcascade_eye.xml'  # haarcascade_eye_tree_eyeglasses.xml  haarcascade_eye
    _neiborhood24 = np.array([[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]],
                            np.uint8)

    rate_h = 1.3
    rate_w = 1.3

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__), self._face_cascade_path))
        self.eye_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__), self._eye_cascade_path))
    # End def

    def change_face_rate(self, rate_h=1.3, rate_w=1.3):
        self.rate_h = rate_h
        self.rate_w = rate_w
    # End def

    def _img_gary(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray
    # End def

    def gamma_compensation(self, img, gamma):
        lookUpTable = np.zeros((256, 1), dtype='uint8')
        for i in range(256):
            lookUpTable[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
        return cv2.LUT(img, lookUpTable)
    # End def

    def contrast_adjustment(self, img, a, b):
        lookUpTable = np.zeros((256, 1), dtype='uint8')
        for i in range(256):
            lookUpTable[i][0] = 255.0 / (1 + math.exp(-a * (i - b) / 255))
        return cv2.LUT(img, lookUpTable)
    # End def

    def detect_face(self, img):
        return_img = None
        img_gray = self._img_gary(img)
        faces = self.face_cascade.detectMultiScale(img_gray)
        for x, y, w, h in faces:
            x = int(x - (self.rate_w - 1.0) * 0.5 * w)
            w = int(h * self.rate_w)
            y = int(y - (self.rate_h - 1.0) * 0.5 * w)
            h = int(h * self.rate_h)
            face = img[y: y + h, x: x + w]
            face_gray = img_gray[y: y + h, x: x + w]
            eyes = self.eye_cascade.detectMultiScale(img_gray)
            if len(eyes) == 2:
                print('======= detect face =========')
                # for (ex, ey, ew, eh) in eyes:
                    # cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                # End for
                # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                img_gamma = self.gamma_compensation(face_gray, 2.0)
                dilated2 = cv2.dilate(img_gamma, self._neiborhood24, iterations=1)
                diff2 = cv2.absdiff(dilated2, img_gamma)
                contour2_tmp = 255 - diff2
                contour2 = self.gamma_compensation(contour2_tmp, 0.5)
                contour3 = self.contrast_adjustment(contour2, 10.0, 128)
                img_contrast = self.contrast_adjustment(face_gray, 10.0, 128)
                dilated4 = cv2.dilate(img_contrast, self._neiborhood24, iterations=1)
                diff4 = cv2.absdiff(dilated4, img_contrast)
                contour4 = 255 - diff4
                return_img = contour3
                return return_img
            # End if
        # End for
        if return_img is None:
            return None
# End class


def main():
    pass
# End def main

if __name__ == '__main__':
    main()
