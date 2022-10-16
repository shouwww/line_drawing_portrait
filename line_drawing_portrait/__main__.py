import cv2
import numpy as np
import PySimpleGUI as sg
import time
from screeninfo import get_monitors
from . import Img_process_tool


def scale_box(img, width, height):
    """指定した大きさに収まるように、アスペクト比を固定して、リサイズする。
    """
    h, w = img.shape[:2]
    aspect = w / h
    if width / height >= aspect:
        nh = height
        nw = int(nh * aspect)
    else:
        nw = width
        nh = int(nw / aspect)
    print(f'w={type(nw)}, h={nw}')
    dst = cv2.resize(img, [nw, nh])

    return dst

def make_sub(x=None, y=None):
    # ------------ サブウィンドウ作成 ------------
    sub_layout = [[sg.Text("サブウィンドウ")],
                [sg.Text("サブウィンドウが現れた！")]
                ]
    return sg.Window("サブウィンドウ", sub_layout, finalize=True, size=(300, 120), location=(x, y))

def main():
    img_tool = Img_process_tool.Img_Process_Tool()
    monitor = get_monitors()[0]
    window_size = (monitor.width, monitor.height)
    print(window_size)

    sg.theme('LightBlue')

    layout = [
             [sg.Text('Realtime movie', size=(20, 1), justification='center', font='Helvetica 20', key='-status-'),
              sg.Text('Camera number: ', size=(8, 1)), sg.InputText(default_text='0', size=(4, 1), key='-camera_num-'),
              sg.Button('Start', size=(10, 1), font='Helvetica 14', key='-start-'),
              sg.Button('Stop', size=(10, 1), font='Helvetica 14', key='-stop-'),
              sg.Button('Exit', size=(10, 1), font='Helvetica 14', key='-exit-'),
              sg.Button('Setting', size=(10, 1), font='Helvetica 14', key='-exit-')],
             [sg.Image(filename='', key='image'), sg.Image(filename='', key='face_image')]]

    window = sg.Window('Realtime movie', layout, size=window_size, location=(0, 0))
    recording = False

    while True:
        # 全てのウィンドウを読み込む
        # window, event, values = sg.read_all_windows()
        event, values = window.read(timeout=20)
        if event in (None, '-exit-'):
            break
        elif event == '-start-':
            window['-status-'].update('Live')
            camera_number = int(values['-camera_num-'])
            cap = cv2.VideoCapture(camera_number)  # , cv2.CAP_DSHOW)
            # cap = cv2.VideoCapture(camera_number)
            recording = True

        elif event == '-stop-':
            window['-status-'].update("Stop")
            recording = False
            # 幅、高さ　戻り値Float
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # print(H,W)
            img = np.full((H, W), 0)
            face_img = np.full((1, 1), 0)
            # ndarry to bytes
            imgbytes = cv2.imencode('.png', img)[1].tobytes()
            window['image'].update(data=imgbytes)
            face_imgbytes = cv2.imencode('.png', face_img)[1].tobytes()
            window['face_image'].update(data=face_imgbytes)
            cap.release()
            cv2.destroyAllWindows()

        if recording:
            ret, frame = cap.read()
            cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # カメラの幅を取得
            cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # カメラの高さを取得
            rate_w = 0.6
            rate_h = 0.5
            if cap_w > monitor.width * rate_w:
                img_w_pix = monitor.width * rate_w
            else:
                img_w_pix = cap_w - 1
            # End if
            if cap_h > monitor.height * rate_h:
                img_h_pix = monitor.width * rate_w
            else:
                img_h_pix = cap_h - 1
            # End if
            print(f'width:{img_w_pix} , height:{img_h_pix}')
            if ret is True:
                set_frame = scale_box(frame, int(img_w_pix), int(img_h_pix))
                imgbytes = cv2.imencode('.png', set_frame)[1].tobytes()
                window['image'].update(data=imgbytes)
                face_img = img_tool.detect_face(set_frame)
                if face_img is not None:
                    face_imgbytes = cv2.imencode('.png', face_img)[1].tobytes()
                    window['face_image'].update(data=face_imgbytes)
                # End if
            # End if
        # End if recording
    window.close()


if __name__ == '__main__':
    main()
