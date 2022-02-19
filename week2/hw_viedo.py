import cv2
#cv2를 로드해오기
import numpy as np
#numpy 로드해오기
net = cv2.dnn.readNetFromTorch('models/eccv16/starry_night.t7')
#이미지 변환 모델을 불러와서 net변수에 할당
cap = cv2.VideoCapture('imgs/03.mp4')
#VideoCapture는 동영상 로드할때 사용하는 메서드.
while True:#무한루
    ret, img = cap.read()
# cap.read()가 img라는 변수에 img를 한장씩 저장해주는 역할을 한다.
    if ret == False:
        break
# ret == Flase의 의미는 동영상이 끝나면 루프를 빠져나와라!
    MEAN_VALUE = [103.939, 116.779, 123.680]
#img처리와 같음으로 한장한장의 img에 MEAN_VALUE 설정
    blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)
#tensor4차원 구조로 바뀐 영상을 blob변수에 할당
    net.setInput(blob)
#추론시작
    output = net.forward()

    output = output.squeeze().transpose((1, 2, 0))
#후처리구문
    output += MEAN_VALUE
    output = np.clip(output, 0, 255)
    output = output.astype('uint8')

    cv2.imshow('result', output)
    if cv2.waitKey(1) == ord('q'):
        break