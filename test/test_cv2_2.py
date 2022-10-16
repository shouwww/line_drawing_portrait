import numpy as np
import math
import cv2


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
    #img = cv2.imread(img_path)
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


cap = cv2.VideoCapture(0)   # USBカメラから映像を撮影、パソコン内蔵カメラの場合は0

print("start")

while True:
    ret, frame = cap.read()  # 1フレーム読み込み
    # 入力画像
    image = frame
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # COLOR_BGR2GRAY , IMREAD_GRAYSCALE
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

    # 実行結果
    cv2.imshow('image ', image)
    cv2.imshow('contour', contour)
    cv2.imshow('contour2', contour2)
    cv2.imshow('contour3', contour3)
    cv2.imshow('contour4', contour4)
    # Canny方によるエッジの検出
    edges = cv2.Canny(contour3, 50, 300)
    cv2.imshow('Canny_contour3', contour3)
    
    if cv2.waitKey(1) != -1:
        # キー入力で終了
        save_edge_points(contour3,"contour3_edge_point.csv")
        break


