import cv2
import numpy as np

proto = 'models/colorization_deploy_v2.prototxt'
weights = 'models/colorization_release_v2.caffemodel'

net = cv2.dnn.readNetFromCaffe(proto, weights)
#net변수에 모델을 할당하는 구문.

pts_in_hull = np.load('models/pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]

net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, np.float32)]
#모델튜닝 과정의 부산물인 코드들.

"""
==============================video_player==================
"""

cap = cv2.VideoCapture('imgs/video.mp4')
#VideoCapture는 동영상 로드할때 사용하는 메서드.

while True:#무한루프
    ret, img = cap.read()
    # cap.read()가 img라는 변수에 img를 한장씩 저장해주는 역할을 한다.
    if ret == False:
        break
    # ret == Flase의 의미는 동영상이 끝나면 루프를 빠져나와라!

    h, w, c = img.shape
    # 높이, 너비 ,채널 을 이미지shape으로 지정

    img_input = img.copy()
    # copy()함수로 저장하기.

    img_input = img_input.astype('float32') / 255.
    # 원래 정수였던 img_input을 float32로 바꿔주는 구문.
    img_lab = cv2.cvtColor(img_input, cv2.COLOR_BGR2Lab)
    # img_input을 BGR 채널에서 lab채널로 바꿔주는 구문.
    img_l = img_lab[:, :, 0:1]
    # 세번째 채널축에서 0번째 채널을 받아오는 구문.

    blob = cv2.dnn.blobFromImage(img_l, size=(224, 224), mean=[50, 50, 50])
    # blobFromImage()함수로 차원변형, resizing, mean value빼주는 구문.

    """
    ======================================추론=========================
    """

    net.setInput(blob)
    # setInput() 함수로 bolb 넣어주기

    output = net.forward()
    # 컴퓨터만 이해하는 결과가 output에 할당되어 있다.
    output = output.squeeze().transpose((1, 2, 0))
    # 사람이 이해할 수 있는 형태로 바꿔주는 구문.

    output_resized = cv2.resize(output, (w, h))
    # img.shape의 w,h//(높이,너비) 비율로 다시 되돌리는 구문.

    output_lab = np.concatenate([img_l, output_resized], axis=2)# <-axis=2 이게 채널방향.
    # l채널 이미지랑 output_resized를 채널방향으로 concatenate(합쳐주는)하는 구문.

    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_Lab2BGR)
    # lab채널을 bgr로 바꿔주는 구문

    output_bgr = output_bgr * 255
    output_bgr = np.clip(output_bgr, 0, 255)
    # 255가 넘는 부분을 잘라내는 구문.

    output_bgr = output_bgr.astype('uint8')
    # float32로 바뀐걸 다시 정수형(int)으로 바꿔주는 구문

    """
    =====================================결과=========================
    """
    cv2.imshow('img', img)
    cv2.imshow('output', output_bgr)

    if cv2.waitKey(1) == ord('q'):
        break