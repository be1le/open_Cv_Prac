import numpy as np
import cv2
from tensorflow.python import tf2
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input

facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
# 얼굴을 판단하는 모델

"""
얼굴영역 탐지모델 ->
입력값 == imgs
출력값 == confidence(확신하는 %정도), 얼굴의 위치(좌표 %로반환)
"""


model = load_model('models/mask_detector.model')
# 마스크를 판단하는 모델
"""
입력값 == 얼굴의 위치(좌표 %로반환)
출력값 == 마스크를 썻는지 안썻는지에 대한 confidence
"""



cap = cv2.VideoCapture(0)
# 부쩍 수척해진 나의 얼굴을 띄우는 구문.

while True:
    ret, img = cap.read()

    if ret == False:
        break
    h, w, c = img.shape
    # 이미지 전처리하기
    blob = cv2.dnn.blobFromImage(img, size=(300, 300), mean=(104., 177., 123.))
    #프레임으로 받아온 이미지를 전처리 하여 blob이라는 변수에 할당 할건데 resize까지 한번에 size=()으로 했다.
    #MEAN_VALUE를 mean=(104., 177., 123.)이렇게 지정함.


    # 얼굴 영역 탐지 모델로 추론하기
    facenet.setInput(blob)
    #전처리한 모델을 setInput으로 지정해 준다.

    dets = facenet.forward()
    #forward()함수를 사용하여 인지한 얼굴들을 dets에 저장시킨다.
    # 각 얼굴에 대해서 반복문 돌기

    #print(dets.shape) -> (1, 1, 200, 7)


    for i in range(dets.shape[2]):#for문으로 돌리는이유 ->사람이 여러명일 수 있어서.
        #(dets.shape[2])이게 왜 얼굴의 개수 -> 모델 설계자 마음!
        confidence = dets[0, 0, i, 2] # i에인지한 인지한 얼굴수

        if confidence < 0.5:
            #model의 confidence가 50%미만으로 나오면 그냥 넘어가라.
            #엄격한 결과를 원할수록 confidence를 증가시키면 된다.
            continue #0.5이하는 continue문으로 그냥 넘어간다.

        # 사각형 꼭지점 찾기 // 모델이 %수치로 반환하기에 원본 이미지의 크기를 곱해주는 구문.
        x1 = int(dets[0, 0, i, 3] * w)#가로를 곱해주기//int()로 소수점이 나오는걸 방지
        y1 = int(dets[0, 0, i, 4] * h)#세로길이를 곱해주기
        x2 = int(dets[0, 0, i, 5] * w)#가로를 곱해주기
        y2 = int(dets[0, 0, i, 6] * h)#세로길이를 곱해주기
        """
        위에서 3번째 인덱스가 왜 3,4,5,6 늘어나게 되는걸까? ->
        해당 tensor 4차원 구조(dets[0, 0, i, 3])에서 3번째 인덱스[0, 0, i, 3<- 이거!]의 3번째 인덱스에!! 
        [0, 0, i, 3<- 이거!] 이array의 3이들어간 자리또한 펼쳐보면 [sum1, sum2, sum3, sum4(이게x1의 좌표)....]
        이런 식으로 array형태 이기에!
         3은 x1 에대응, 4는 y1에대응, 5는 x2에 대응, 6은 y2에 대응 되기 때문입니다! 
        """
        #얼굴영억 출력값(얼굴인식 모델에서 파생된)을 전처리 하는 구문
        face = img[y1:y2, x1:x2]
        #얼굴의 위치를 face라는 변수에 할당.


        face_input = cv2.resize(face, dsize=(224, 224))
        # 리사이징하는 구문

        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        #cvtColor()로 RGB로 바꿔준다. 해당모델이 RGB로 학습되어 있기 때문.

        face_input = preprocess_input(face_input)
        #preprocess_input()함수로 전처리.
        #(224,224,3)

        face_input = np.expand_dims(face_input, axis=0)
        #expand_dims()함수로 (1,224,224,3) tensor4차원 구조로 차원변형.

        mask, nomask = model.predict(face_input).squeeze()
        #둘이합쳐서 컨피던스가 1이되도록 만드는 구문. squeeze()함수로 변형된 차원 후처리.


        if mask > nomask:#마스크를 썼으면 파란색으로 그릴게.
            color = (0, 255, 0)
            label = 'Mask %d%%' % (mask * 100)
            #소수점 2번째 자리까지 Confidence를 출력하기 위한 구문

        else:#마스크를 안썻으면 공주핑크로 그릴게.
            color = (153, 51, 255)
            label = 'No Mask %d%%' % (nomask * 100)
            # 소수점 2번째 자리까지 Confidence를 출력하기 위한 구문

        # 사각형 그리기
        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color)#<-위의 color


        cv2.putText(img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color,
                    thickness=2)
        # putText()함수로 글씨 작성,text = '%s, %s'를 (gender, age)순으로 띄워주겠다.
        # org=(x1, y1)얼굴의 위치
        # font 부분은 글씨체 설정 구문


    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break
