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
    _nose_cascade_path = '../libs/haarcascades/haarcascade_mcs_nose.xml'
    _neiborhood24 = np.array([[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]],
                            np.uint8)

    rate_h = 1.3
    rate_w = 1.3
    canny_threshold = [300, 300]

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__), self._face_cascade_path))
        self.eye_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__), self._eye_cascade_path))
        self.nose_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__), self._nose_cascade_path))
    # End def

    def change_face_rate(self, rate_h=1.3, rate_w=1.3):
        self.rate_h = rate_h
        self.rate_w = rate_w
    # End def

    def change_canny_threshold(self, threshold1, threshold2):
        self.canny_threshold = [threshold1, threshold2]
    # End def

    def get_canny_threshold(self):
        return self.canny_threshold
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
                # Canny方によるエッジの検出
                face_gray = cv2.GaussianBlur(face_gray, ksize=(3, 3), sigmaX=1.3)
                return_img = cv2.Canny(face_gray, self.canny_threshold[0], self.canny_threshold[1])

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
