import numpy as np
import cv2
import datetime
import draw_main

# https://www.tech-teacher.jp/blog/python-opencv/ こっちの方が良いかも

# 撮影時間を秒で指定  3分の場合は180
time = 10

cap = cv2.VideoCapture(0)   # USBカメラから映像を撮影、パソコン内蔵カメラの場合は0
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # カメラの幅を取得
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # カメラの高さを取得

# fps × 指定した撮影時間の値繰り返す
print("start")
while True:
    ret, frame = cap.read()  # 1フレーム読み込み
    # 入力画像
    image = frame
    # image = cv2.imread('input/cup.jpg')

    # 画像のサイズ縮小
    height = image.shape[0]
    width = image.shape[1]
    # image = cv2.resize(image, (width, height))

    # Haar-like特微分類器の読み込み
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

    # gary scale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ditect_face
    faces = face_cascade.detectMultiScale(img_gray)

    ''' # 輪郭抽出
    param_edge = 107
    ret, thresh = cv2.threshold(img_gray, param_edge, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(image, contours, -1, (0, 0, 255), 3)   # 全輪郭を描画
    '''

    # Canny方によるエッジの検出
    edges = cv2.Canny(image, 50, 300)

    rate_h = 1.3
    rate_w = 1.3
    # draw_detect area
    for x, y, w, h in faces:
        x = int(x - (rate_w - 1.0) * 0.5 * w)
        w = int(h * rate_w)
        y = int(y - (rate_h - 1.0) * 0.5 * w)
        h = int(h * rate_h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(edges, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = image[y: y + h, x: x + w]
        face_edges = edges[y: y + h, x: x + w]
        face_gray = img_gray[y: y + h, x: x + w]
        eyes = eye_cascade.detectMultiScale(face_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.rectangle(face_edges, (ex, ey), (ex + ew, ey + eh), (255, 255, 255), 2)

    # 実行結果
    cv2.imshow('image ', image)
    cv2.imshow('edges', edges)
    # cv2.imshow('Original', img_gray)
    if cv2.waitKey(1) != -1:
        out_path = "cama_edges.csv"
        # エッジになっているx,y座標を取り出す
        h, w = edges.shape
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        # 255になっている部分がエッジ部分
        X_true = X[edges > 128]
        Y_true = Y[edges > 128]
        # エッジの点になっている座標が入っている
        index = np.array([X_true, Y_true]).T
        # 保存
        f = open(out_path, "w")
        f.write("x,y\n")
        for i in range(len(index)):
            f.write(str(index[i, 0]) + "," + str(index[i, 1]) + "\n")
        # End for
        f.close()
        tsp = draw_main.TSP(path="cama_edges.csv", alpha=1.0, beta=16.0, Q=1.0e3, vanish_ratio=0.8)
        tsp.solve(100)
        tsp.save("best_order_cam.csv")
        tsp.plot(tsp.result)
        # キー入力で終了
        break

print("stop")
cap.release()
cv2.destroyAllWindows()
