import numpy as np
import math
import cv2
import csv
import time


def gamma_compensation(img, gamma):
    lookUpTable = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        lookUpTable[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
    return cv2.LUT(img, lookUpTable)
# End def


def contrast_adjustment(img, a, b):
    lookUpTable = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        lookUpTable[i][0] = 255.0 / (1 + math.exp(-a * (i - b) / 255))
    return cv2.LUT(img, lookUpTable)
# End def


def scale_box(img, width, height):
    """指定した大きさに収まるように、アスペクト比を固定して、リサイズする。
    """
    h, w = img.shape[:2]
    aspect = w / h
    if width / height >= aspect:
        nh = height
        nw = round(nh * aspect)
    else:
        nw = width
        nh = round(nw / aspect)

    dst = cv2.resize(img, dsize=(nw, nh))

    return dst


def save_edge_points(img, out_path):
    # 画像を読み込んでエッジ処理
    # img = cv2.imread(img_path)
    edge = cv2.Canny(img, 100, 200)

    # エッジになっているx,y座標を取り出す
    h, w = edge.shape
    x = np.arange(w)
    y = np.arange(h)

    X, Y = np.meshgrid(x, y)

    # 255になっている部分がエッジ部分
    X_true = X[edge > 128]
    Y_true = Y[edge > 128]

    # エッジの点になっている座標が入っている
    index = np.array([X_true, Y_true]).T

    # 保存
    f = open(out_path, "w")
    f.write("x,y\n")
    for i in range(len(index)):
        f.write(str(index[i, 0]) + "," + str(index[i, 1]) + "\n")
    # End for
    f.close()
# End for

data_xy = []
# 入力画像
image = cv2.imread('test_img.png')
# change gray scale 
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # COLOR_BGR2GRAY , IMREAD_GRAYSCALE
img_white = np.ones(image.shape, np.uint8) * 255
# Thresholding
#ret, thresh = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY)
#ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow('Thresholding', thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
thresh = cv2.dilate(thresh, kernel)
#thresh = cv2.erode(thresh, kernel)
cv2.imshow('drawContours', thresh)
# cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
# 輪郭検出 （cv2.ChAIN_APPROX_SIMPLE） cv2.CHAIN_APPROX_NONE
contours1, hierarchy1 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for i,data in enumerate(contours1):
    if cv2.contourArea(data) < 500:
        print(f'pass , i={i} , area={cv2.contourArea(data)}')
    else:
        start_point = [0,0]
        end_point = [0,0]
        for point in data:
            #print(point)
            #print(f'point_x:{point[0][0]} , point_y:{point[0][1]}')
            #print(len(point))
            #print(type(point))
            data_xy.append([point[0][0], point[0][1]])
            end_point = [point[0][0], point[0][1]]
            if start_point ==[0,0]:
                pass
            else:
                img_white = cv2.line(img_white,(start_point[0],start_point[1]),(end_point[0],end_point[1]),(125,125,125),1)
                cv2.imshow('draw_line',img_white)
                # cv2.waitKey(1)
            start_point = end_point
        # End for
    # End if
# End for
#輪郭の描画
# cv2.drawContours(img_white, contours1, -1, (125, 125, 125), 1, cv2.LINE_AA)
img_canny = cv2.Canny(thresh, 100, 300)
cv2.imshow('image', image)
cv2.imshow('gray', img_gray)
cv2.moveWindow('image', 100, 200)
cv2.moveWindow('gray', 300, 200)
cv2.moveWindow('Thresholding', 500, 200)
cv2.moveWindow('drawContours', 700, 200)
cv2.imwrite('img.png',thresh)

"""
neiborhood24 = np.array([[1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1]],
                        np.uint8)
dilated = cv2.dilate(img_gray, neiborhood24, iterations=1)
diff = cv2.absdiff(dilated, img_gray)
contour = 255 - diff

img_gamma = gamma_compensation(img_gray, 2.0)
dilated2 = cv2.dilate(img_gamma, neiborhood24, iterations=1)
diff2 = cv2.absdiff(dilated2, img_gamma)
contour2_tmp = 255 - diff2
contour2 = gamma_compensation(contour2_tmp, 0.5)

contour3 = contrast_adjustment(contour2, 10.0, 128)

img_contrast = contrast_adjustment(img_gray, 10.0, 128)
dilated4 = cv2.dilate(img_contrast, neiborhood24, iterations=1)
diff4 = cv2.absdiff(dilated4, img_contrast)
contour4 = 255 - diff4
"""

# 実行結果
#cv2.imshow('image ', image)
#cv2.imshow('contour', contour)
#cv2.imshow('contour2', contour2)
#cv2.imshow('contour3', contour3)
#cv2.imshow('contour4', contour4)
# Canny方によるエッジの検出
#edges = cv2.Canny(contour3, 100, 300)
#cv2.imshow('Canny_contour3', image)

print(len(data_xy))
f = open('out2.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerows(data_xy)
f.close()

cv2.waitKey()