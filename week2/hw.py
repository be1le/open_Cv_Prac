import cv2
import numpy as np

"""
========================================flow======================================
| Preprocessing(전처리) -> Inference(추론) -> Postprocessing(후처리) -> Print(출력)  |
==================================================================================
"""


net = cv2.dnn.readNetFromTorch('models/instance_norm/candy.t7')
#open cv의 dnn으로 토치로부터 모델을 읽어내겠다. 그리고 net이라는 변수에 할당하겠다.


img = cv2.imread('imgs/hw.jpg')
#img읽어오는 구문


cropped_img = img[140:370, 480:810]
#이미지를 cropped하는 구문

#=======================================전처리==============================
#전처리를 시작하는 구문
h, w, c = cropped_img.shape
#이미지형태 h높이 w너비 c차원


img = cv2.resize(img, dsize=(1600, int(h / w * 1600)))
#가로500과 500에맞는 세로의 크기로 넣어주겠다.
#(h / w * 500)의 값이 소수점일수 있기에 int()를사용하여 정수로 반환한다.
#원본이미지의 비율을 망가트리지 않으면서 사용하고 싶기 때문에 방정식 사용.

img = img[185:457, 606:1007]

print(img.shape)#(325, 500, 3)
#높이325,너비500,채널이3인 shape에서
#3d tensor 구조

MEAN_VALUE = [103.939, 116.779, 123.680]
blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)
#blobFromImage 전처리는 함수이다. meanvalue는 가이드라인을 따르는 값 // 가장결과가 좋은 값이다.
#blobFromImage가 차원변형을 해준다.

print(blob.shape)#(1, 3, 325, 500)
#픽셀의값은 변하지 않았지만, 약간의 순서가 변했다. ->컴퓨터가 알아들을 수 있게 바꿔준다.
#텐서플로우에서는 채널이 맨앞에 정의되어 있어야 하기 때문에 채널을 앞으로 보내주었다.

"""
딥러닝에서의 텐서 구조에 대하여 
입력 모양은 (N, H, W, C)이고 우리는 출력 모양은 (N, C, H, W).
따라서 tf.transpose잘 선택된 순열 로 적용해야 한다.
맨앞의 1은 batch size 여기서는 한개씩 넣어주겠다.
4d tensor 구조(컴퓨터의 입장에선 3d)
"""


#===============================추론===============================
net.setInput(blob)
#전처리한결과를 인풋으로 지정해 주는 구문
output = net.forward()
#Inference하는 구문 //추론문



#==============================후처리=============================
#후처리가 시작하는 구문
output = output.squeeze().transpose((1, 2, 0))
#squeeze()함수로 blobFromImage()함수로 늘린 차원을 다시 줄여준다.
#transpose() 차원변형을 반대로 해주는 함수이다.


output += MEAN_VALUE
#전처리할때 빼줬던 MEAN_VALUE 를 다시 더해준다.

output = np.clip(output, 0, 255)
#np.clip() 함수로 MEAN_VALUE를 더했을때 255가 넘는걸 다255로 제한하겠다.


#==========================출력==================================
output = output.astype('uint8')
#정수형태로 바뀌면서 사람이 볼수있는 이미지 형태로 만들어 준다.


cv2.imshow('output',output)

cv2.imshow('imgs',img)#앞에있는 'imgs'라는이름의 창으로 변수에할당된 img를 보여줘~ 라는 뜻
cv2.waitKey(0)#wait key==0 계속띄워놔, 계속기다려줘

