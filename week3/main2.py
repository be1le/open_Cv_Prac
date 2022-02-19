import numpy as np
import cv2

facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')

gender_list = ['Male', 'Female']# 0-> 남자, 1 ->여자
age_list = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
           #   0        1        2          3         4           5         6         7
           # 이런 식으로 나이가 범위로 분류되어 인덱스 값으로 반환된다.

gender_net = cv2.dnn.readNetFromCaffe('models/deploy_gender.prototxt', 'models/gender_net.caffemodel')
#성별을 구분해 주는 모델


age_net = cv2.dnn.readNetFromCaffe('models/deploy_age.prototxt', 'models/age_net.caffemodel')
#나이를 구분해 주는 모델.


img = cv2.imread('imgs/02.jpg')

h, w, c = img.shape

# 이미지 전처리하기
blob = cv2.dnn.blobFromImage(img, size=(300, 300), mean=(104., 177., 123.))

# 얼굴 영역 탐지 모델로 추론하기
facenet.setInput(blob)
# 전처리한 모델을 setInput으로 지정해 준다.

dets = facenet.forward()
# print(dets.shape) -> (1, 1, 200, 7)

# 각 얼굴에 대해서 반복문 돌기
for i in range(dets.shape[2]):#for문으로 돌리는이유 ->사람이 여러명일 수 있어서.
    # (dets.shape[2])이게 왜 얼굴의 개수 -> 모델 설계자 마음!
    confidence = dets[0, 0, i, 2] #i에인지한 인지한 얼굴수


    if confidence < 0.5:
        # model의 confidence가 50%미만으로 나오면 그냥 넘어가라.
        # 엄격한 결과를 원할수록 confidence를 증가시키면 된다.
        continue #0.5이하는 continue문으로 그냥 넘어간다.

    # 사각형 꼭지점 찾기 // 모델이 %수치로 반환하기에 원본 이미지의 크기를 곱해주는 구문.
    x1 = int(dets[0, 0, i, 3] * w)  # 가로를 곱해주기//int()로 소수점이 나오는걸 방지
    y1 = int(dets[0, 0, i, 4] * h)  # 세로길이를 곱해주기
    x2 = int(dets[0, 0, i, 5] * w)  # 가로를 곱해주기
    y2 = int(dets[0, 0, i, 6] * h)  # 세로길이를 곱해주기


# 위에서 3번째 인덱스가 왜 3,4,5,6 늘어나게 되는걸까? ->
# 해당 tensor 4차원 구조(dets[0, 0, i, 3])에서 3번째 인덱스[0, 0, i, 3<- 이거!]의 3번째 인덱스에!!
# [0, 0, i, 3<- 이거!] 이array의 3이들어간 자리또한 펼쳐보면 [sum1, sum2, sum3, sum4(이게x1의 좌표)....]
# 이런 식으로 array형태 이기에!
#  3은 x1 에대응, 4는 y1에대응, 5는 x2에 대응, 6은 y2에 대응 되기 때문입니다!
#  3주차 imgs 폴더 explain+에 사진설명 첨부.


    face = img[y1:y2, x1:x2]
    # 얼굴의 위치를 face라는 변수에 할당.


    blob = cv2.dnn.blobFromImage(face, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746))
    #위의 구문에서 리사이징,차원변형,민값도 빼주는 구문

    gender_net.setInput(blob)
    #setInpuy() ->Net에 blob 형태의 데이터 넣어주는 함수

    gender_index = gender_net.forward().squeeze().argmax()
    #추론구문.
    #forward() -> Net을 실행 시켜줌(순방향)
    # squeeze() -> 차원 축소
    # argmax() ->
    """ age_list = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
                      0          1        2          3         4          5          6          7
     age_list 에서 해당 img의 예측결과를 index로 뽑아아와주는 함수."""

    gender = gender_list[gender_index]
    #나온 gender inedex에서 list를 뽑아내는 구문.

    age_net.setInput(blob)
    # setInpuy() ->Net에 blob 형태의 데이터 넣어주는 함수

    age_index = age_net.forward().squeeze().argmax()
    # 추론구문.
    # forward() -> Net을 실행 시켜줌(순방향)
    # squeeze() -> 차원 축소
    # argmax() ->
    """ age_list = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
                      0          1        2          3         4          5          6          7
     age_list 에서 해당 img의 예측결과를 index로 뽑아아와주는 함수."""

    age = age_list[age_index]
    #age라는 변수에 그결과를 할당


    cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=2)
    # 사각형 그리기 구문.

    cv2.putText(img, '%s, %s' % (gender, age), org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(0, 255, 0), thickness=2)
    #putText()함수로 글씨 작성,text = '%s, %s'를 (gender, age)순으로 띄워주겠다.
    # org=(x1, y1)얼굴의 위치
    #font 부분은 글씨체 설정 구문, color 는 위에서 b가 255라서 파란색으로 뜨게됨.


cv2.imshow('result', img)
#이미지를 'result'라는 창에서 띄워줘.
cv2.waitKey(0)
#계속 띄워놔줘.