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
edges = np.ones((100,100), np.uint8) * 255

set_param_flg = False
threshold1 = 400
threshold2 = 400
while True:
    ret, frame = cap.read()  # 1フレーム読み込み
    # 入力画像
    image = frame
    #image = cv2.imread('test_img1_out.png')

    # 画像のサイズ縮小
    height = image.shape[0]
    width = image.shape[1]
    # image = cv2.resize(image, (width, height))

    # Haar-like特微分類器の読み込み
    face_cascade = cv2.CascadeClassifier('../libs/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('../libs/haarcascades/haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier('../libs/haarcascades/haarcascade_mcs_nose.xml')
    

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

    

    rate_h = 1.3
    rate_w = 1.3
    #edges = cv2.Canny(img_gray, threshold1, threshold2)    
    # draw_detect area
    for x, y, w, h in faces:
        x = int(x - (rate_w - 1.0) * 0.5 * w)
        w = int(h * rate_w)
        y = int(y - h) #  int(y - (rate_h - 1.0) * 0.5 * w)
        h = int(h * rate_h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #cv2.rectangle(edges, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = image[y: y + h, x: x + w]
        #face_edges = edges[y: y + h, x: x + w]
        face_gray = img_gray[y: y + h, x: x + w]
        eyes = eye_cascade.detectMultiScale(face_gray)
        noses = nose_cascade.detectMultiScale(face_gray, 1.3, 5) # 検出の実行
        if len(eyes) == 2:
            # Canny方によるエッジの検出
            face_gray = cv2.GaussianBlur(face_gray, ksize=(3, 3), sigmaX=1.3)
            edges = cv2.Canny(face_gray, threshold1, threshold2)
        # End if
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
            #cv2.rectangle(face_edges, (ex, ey), (ex + ew, ey + eh), (255, 255, 255), 2)
        # End for
        for (nx,ny,nw,nh) in noses:
            cv2.rectangle(face, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 1)
        # End for
    # 実行結果
    cv2.imshow('image ', image)
    cv2.imshow('edges', edges)
    # cv2.imshow('Original', img_gray)
    key_input = cv2.waitKey(1)
    if key_input == 27:
        img_white = np.ones(edges.shape, np.uint8) * 255
        data_xy=[]
        print(key_input)
        contours1, hierarchy1 = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for i,data in enumerate(contours1):
            aaa = cv2.contourArea(data)
            if aaa < 100:
                print(f'pass , i={i} , area={aaa}, points={len(data)}')
            else:
                print(f'==draw , i={i} , area={aaa}, points={len(data)}')
                start_point = [0,0]
                end_point = [0,0]
                for point in data:
                    data_xy.append([point[0][0], point[0][1]])
                    end_point = [point[0][0], point[0][1]]
                    if start_point ==[0,0]:
                        pass
                    else:
                        img_white = cv2.line(img_white,(start_point[0],start_point[1]),(end_point[0],end_point[1]),(125,125,125),1)
                        # print(f'write [{start_point[0]}][{start_point[1]}] , [{end_point[0]}][{end_point[1]}]')
                        cv2.imshow('draw_line',img_white)
                        cv2.waitKey(1)
                    # End if
                    start_point = end_point
                # End for
            # End if
        # End for
        # キー入力で終了
        print('finish')
        cv2.waitKey(0)
        break
    elif key_input == ord('a'):
        threshold1 = threshold1 -10
        set_param_flg = True
    elif key_input == ord('d'):
        threshold1 = threshold1 +10
        if threshold1 > threshold2:
            threshold2 = threshold1
        # End if
        set_param_flg = True
    elif key_input == ord('s'):
        threshold2 = threshold2 -10
        if threshold1 > threshold2:
            threshold1 = threshold2
        # End if
        set_param_flg = True
    elif key_input == ord('w'):
        threshold2 = threshold2 +10
        set_param_flg = True
    # End if
    if set_param_flg:
        print(f'low={threshold1} , high={threshold2}')
        set_param_flg = False

print("stop")
cap.release()
cv2.destroyAllWindows()
