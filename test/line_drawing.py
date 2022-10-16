import cv2
import datetime
import numpy as np
import math


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

# fpsを20.0にして撮影したい場合はfps=20.0にします
fps = 30.0
cap = cv2.VideoCapture(0)   # USBカメラから映像を撮影、パソコン内蔵カメラの場合は0
cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # カメラの幅を取得
cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # カメラの高さを取得

face_cascade_path = '../libs/haarcascades/haarcascade_frontalface_default.xml'
eye_cascade_path = '../libs/haarcascades/haarcascade_eye.xml'  # haarcascade_eye_tree_eyeglasses.xml  haarcascade_eye
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# fps × 指定した撮影時間の値繰り返す
print("start")
while True:
    ret, frame = cap.read()  # 1フレーム読み込み
    # 入力画像
    img = frame
    # グレースケール化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    neiborhood24 = np.array([[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]],
                            np.uint8)

    rate_h = 1.3
    rate_w = 1.3
    faces = face_cascade.detectMultiScale(img_gray)
    for x, y, w, h in faces:
        x = int(x - (rate_w - 1.0) * 0.5 * w)
        w = int(h * rate_w)
        y = int(y - (rate_h - 1.0) * 0.5 * w)
        h = int(h * rate_h)
        face = img[y: y + h, x: x + w]
        face_gray = img_gray[y: y + h, x: x + w]
        eyes = eye_cascade.detectMultiScale(img_gray)
        if len(eyes) == 2:
            print('======= detect face =========')
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            # End for
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            img_gamma = gamma_compensation(face_gray, 2.0)
            dilated2 = cv2.dilate(img_gamma, neiborhood24, iterations=1)
            diff2 = cv2.absdiff(dilated2, img_gamma)
            contour2_tmp = 255 - diff2
            contour2 = gamma_compensation(contour2_tmp, 0.5)
            contour3 = contrast_adjustment(contour2, 10.0, 128)
            img_contrast = contrast_adjustment(face_gray, 10.0, 128)
            dilated4 = cv2.dilate(img_contrast, neiborhood24, iterations=1)
            diff4 = cv2.absdiff(dilated4, img_contrast)
            contour4 = 255 - diff4
            cv2.imshow('face image gray', contour3)
            break
        # End if
    # End for

    # 閾値処理
    #ret,thresh = cv2.threshold(img_gray,95,255,cv2.THRESH_BINARY)
    # 輪郭検出 （cv2.ChAIN_APPROX_SIMPLE）
    #contours1, hierarchy1 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 輪郭の描画
    #cv2.drawContours(img, contours1, -1, (0, 255, 0), 2, cv2.LINE_AA)
    # 実行結果
    cv2.imshow('source image', img)
    # cv2.imshow('Original', img_gray)
    if cv2.waitKey(1) != -1:
        # キー入力で終了
        break

print("stop")
cap.release()
cv2.destroyAllWindows()
