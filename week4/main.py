import cv2
#cv2를 로드해오기

import dlib
#dlib를 로드해오기

detector = dlib.get_frontal_face_detector()
#detector변수에 얼굴 영역을 탐지하는 모델 로드후 할당.


cap = cv2.VideoCapture('videos/01.mp4')
#VideoCapture는 동영상 로드할때 사용하는 메서드.

sticker_img = cv2.imread('imgs/sticker01.png', cv2.IMREAD_UNCHANGED)
#cv2.IMREAD_UNCHANGED가 있어야 투명도가 포함된 채널이 생성된다.


while True:#무한루프
    ret, img = cap.read()
#cap.read()가 img라는 변수에 img를 한장씩 저장해주는 역할을 한다.
    if ret == False:
        break
#ret == Flase의 의미는 동영상이 끝나면 루프를 빠져나와라!

    dets = detector(img)#이미지를 detector에 넣는 구문.
    #함수내에 이미지 전처리 기능이 있어서 전처리가 필요하지 않다.
    print("number of faces detected:", len(dets))
    #dets -> 사람얼굴이 리스트 형태로 저장되어 있다.
    #len()함수를 사용하여 리스트의 길이 == 사람수를 반환한다.

    for det in dets:
        x1 = det.left() - 40
        y1 = det.top() - 50
        x2 = det.right() + 40
        y2 = det.bottom() + 30
    #두점의 좌표를 받아와서 네모를 그려주는 모델이라 2점의 위치를 구하는 구문
    # cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=  주석해재시 네모를 그려줌.
    # (153, 51, 255),thickness=2)  #남자는 공주핑크


        try:   #스티커 이미지가 화면밖에있을때 나는 에러 방지를 위한 try except문
            overlay_img = sticker_img.copy()
            #sticker_img = cv2.imread('imgs/sticker01.png', cv2.IMREAD_UNCHANGED)
            #이구문에서 복사된 스티커를 카피해서 overlay_img변수에 할당

            overlay_img = cv2.resize(overlay_img, dsize=(x2 - x1, y2 - y1))
            #스티커를 얼굴크기에 맞게 리사이징 하는 구문.
            #(x2 - x1, y2 - y1) <- 이구문으로 얼굴크기에 맞게 조정.

            overlay_alpha = overlay_img[:, :, 3:4] / 255.0
            background_alpha = 1.0 - overlay_alpha
            img[y1:y2, x1:x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[y1:y2, x1:x2]
        except:
            pass



    cv2.imshow('result', img)
#result에 저장된 이미지자료를 읽어올거니까 imshow사용
    if cv2.waitKey(1) == ord('q'):
#이미지당 1ms기다리면서 넘기고 키보드 q 누르면 동영상을 꺼라
#waitKey() <-내부의 숫자가 커질수록 천천히 재생된다. 오래기다렸다가 다음이미지를 보여주니까.
        break
