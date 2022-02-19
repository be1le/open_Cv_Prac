import cv2
#cv2를 로드해오기

import dlib
#dlib를 로드해오기


detector = dlib.get_frontal_face_detector()
#detector변수에 얼굴 영역을 탐지하는 모델 로드후 할당.

predictor = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')
#렌드마크를 찾아주는 모델 로드후 할당.

cap = cv2.VideoCapture('videos/01.mp4')
#VideoCapture는 동영상 로드할때 사용하는 메서드.

sticker_img = cv2.imread('imgs/gangster.png', cv2.IMREAD_UNCHANGED)
#cv2.IMREAD_UNCHANGED가 있어야 투명도가 포함된 채널이 생성된다.

sticker_img_1 = cv2.imread('imgs/weeed.png')
#cv2.IMREAD_UNCHANGED가 있어야 투명도가 포함된 채널이 생성된다.

while True:#무한루프
    ret, img = cap.read()
#cap.read()가 img라는 변수에 img를 한장씩 저장해주는 역할을 한다.
    if ret == False:
        break
#ret == Flase의 의미는 동영상이 끝나면 루프를 빠져나와라!

    dets = detector(img)#이미지를 detector에 넣는 구문.
    #함수내에 이미지 전처리 기능이 있어서 전처리가 필요하지 않다.
    #dets -> 사람얼굴이 리스트 형태로 저장되어 있다.
    #len()함수를 사용하여 리스트의 길이 == 사람수를 반환한다.

    for det in dets:#여러개의 얼굴영억(dets)에서 한개씩(det) 돌아가면서
        shape = predictor(img, det)
        #전체 이미지(img)와 얼굴영역 좌표(det)넣어주는 구문.
        try: #스티커 이미지가 화면밖에있을때 나는 에러 방지를 위한 try except문
            x1 = det.left()
            y1 = det.top()
            x2 = det.right()
            y2 = det.bottom()
            glasses_x1 = shape.parts()[2].x - 35 #<- 왼쪽 눈꼬리
            glasses_x2 = shape.parts()[0].x + 35#<- 오른쪽 눈꼬리

            h, w, c = sticker_img.shape
            #sticker_img.shape의 (높이,너비,채널) <- shape을 알아보는 구문.

            glasses_w = glasses_x2 - glasses_x1 # <- 가로길이
            glasses_h = int(h / w * glasses_w) # <- 세로길이는 비례식 사용!

            center_y = (shape.parts()[0].y + shape.parts()[2].y) / 2
            #안경의 중심좌표를 구하는 구문.

            glasses_y1 = int(center_y - glasses_h / 2)
            glasses_y2 = glasses_y1 + glasses_h
            #추가적인 안경의 위치를 구하는 구문.

            overlay_img = sticker_img.copy()
            overlay_img = cv2.resize(overlay_img, dsize=(glasses_w, glasses_h))

            overlay_alpha = overlay_img[:, :, 3:4] / 255.0
            background_alpha = 1.0 - overlay_alpha

            img[glasses_y1:glasses_y2, glasses_x1:glasses_x2] = overlay_alpha * overlay_img[:, :,:3] \
            + background_alpha * img[glasses_y1:glasses_y2,glasses_x1:glasses_x2]


            #=============================================================

            a1 = det.left()
            b1 = det.top()
            a2 = det.right()
            b2 = det.bottom()
            weeed_x1 = shape.parts()[4].y - 30   # <- 코의 좌표
            weeed_x2 = shape.parts()[3].y  # <- 오른쪽 눈꼬리

            hh, ww, cc = sticker_img_1.shape
            # sticker_img.shape의 (높이,너비,채널) <- shape을 알아보는 구문.

            weeed_ww = weeed_x2 -  weeed_x1  # <- 가로길이
            weeed_hh = int(hh / ww * weeed_ww)  # <- 세로길이는 비례식 사용!

            center_yy = (shape.parts()[0].y + shape.parts()[2].y) / 2
            # 떠그의 중심좌표를 구하는 구문.

            weeed_y1 = int( center_yy - weeed_hh / 2)
            weeed_y2 = weeed_y1 + weeed_hh
            # 추가적인 안경의 위치를 구하는 구문.

            overlay_img_1 = sticker_img_1.copy()
            overlay_img_1 = cv2.resize(overlay_img_1, dsize=(weeed_ww, weeed_hh))

            overlay_alpha_1 = overlay_img_1[:, :, 3:4] / 255.0
            background_alpha_1 = 1.0 - overlay_alpha_1

            img[weeed_y1:weeed_y2,  weeed_x1: weeed_x2] = overlay_alpha_1 * overlay_img_1[:, :, :3] \
            + background_alpha_1 * img[weeed_y1:weeed_y2, weeed_x1:weeed_x2]


        except:
            pass




    cv2.imshow('result', img)
    #result에 저장된 이미지자료를 읽어올거니까 imshow사용
    if cv2.waitKey(1) == ord('q'):
    #이미지당 1ms기다리면서 넘기고 키보드 q 누르면 동영상을 꺼라
    #waitKey() <-내부의 숫자가 커질수록 천천히 재생된다. 오래기다렸다가 다음이미지를 보여주니까.
        break
