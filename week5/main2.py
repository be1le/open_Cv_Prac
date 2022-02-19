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


#이미지 전처리 시작하는 구문.
img = cv2.imread('imgs/01.jpg')
#imread로 읽어오는 구문

h, w, c = img.shape
#높이, 너비 ,채널 을 이미지shape으로 지정

img_input = img.copy()
# copy()함수로 저장하기.

img_input = img_input.astype('float32') / 255.
#원래 정수였던 img_input을 float32로 바꿔주는 구문.

img_lab = cv2.cvtColor(img_input, cv2.COLOR_BGR2Lab)
#img_input을 BGR 채널에서 lab채널로 바꿔주는 구문.

img_l = img_lab[:, :, 0:1]
#세번째 채널축에서 0번째 채널을 받아오는 구문.


blob = cv2.dnn.blobFromImage(img_l, size=(224, 224), mean=[50, 50, 50])
#blobFromImage()함수로 차원변형, resizing, mean value빼주는 구문.


"""
======================================추론=========================
"""

net.setInput(blob)
#setInput() 함수로 bolb 넣어주기.

output = net.forward()
#컴퓨터만 이해하는 결과가 output에 할당되어 있다.
"""
======================================후처리=============================
"""

output = output.squeeze().transpose(1, 2, 0)
#사람이 이해할 수 있는 형태로 바꿔주는 구문.

output_resized = cv2.resize(output, (w,h))
#img.shape의 w,h//(높이,너비) 비율로 다시 되돌리는 구문.

output_lab = np.concatenate([img_l, output_resized], axis=2) # <-axis=2 이게 채널방향.
#l채널 이미지랑 output_resized를 채널방향으로 concatenate(합쳐주는)하는 구문.

output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_Lab2BGR)
#lab채널을 bgr로 바꿔주는 구문

output_bgr = output_bgr * 255
output_bgr = np.clip(output_bgr, 0, 255)
#255가 넘는 부분을 잘라내는 구문.

output_bgr = output_bgr.astype('uint8')
#float32로 바뀐걸 다시 정수형(int)으로 바꿔주는 구문

mask = np.zeros_like(img, dtype='uint8')
#이미지와 같은 형태를 유지하면서 0으로(검은색)채운 이미지를 만들어주는 구문.

mask = cv2.circle(mask, center=(260, 260), radius=200, color=(1, 1, 1), thickness=-1)# <-안쪽 채우기.
#circle()함수를 사용하여 마스크를 원으로 그리기

color = output_bgr * mask #<-마스크를 한 부분.
gray = img * (1 - mask) #<- 마스크를 안한 부분.
#(1 - mask)식을 이용하여 마스크부분과 나머지를 반전시켜주는 구문

output2 = color + gray
#마스크된부분과 나머지영역을 더해서 같이 볼 수 있게 만드는 구문

cv2.imshow('result2', output2)


"""
=====================================결과=========================
"""
cv2.imshow('img', img_input)
#input이미지를 띄워주는 구문

cv2.imshow('result', output_bgr)
#output이미지를 띄워주는 구문

cv2.waitKey(0)