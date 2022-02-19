import cv2
#cv2를 로드해오기
cap = cv2.VideoCapture('04.mp4')# <- 여기를 0으로 바꾸면 본인의 웹캠으로 연결된다.
#VideoCapture는 동영상 로드할때 사용하는 메서드.
while True:#무한루프
    ret, img = cap.read()
#cap.read()가 img라는 변수에 img를 한장씩 저장해주는 역할을 한다.
    if ret == False:
        break
# ret == Flase의 의미는 동영상이 끝나면 루프를 빠져나와라!

    cv2.rectangle(img, pt1=(721, 183), pt2=(878,465), color=(255, 0, 0),thickness=2)
#동영상에 사각형을 그려보는 구문.

    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#이지미를 그레이 스케일로 변환하는 구문

    img = cv2.resize(img, dsize=(640, 360))
#이미지를 리사이징 해보는 구문

    img = img[100:200, 150:250]
# #이미지를 자르는 구문 // 리사이징과 엄연히 다른 개념이다.




    cv2.imshow('result', img)
#result에 저장된 이미지자료를 읽어올거니까 imshow사용
    if cv2.waitKey(1) == ord('q'):
#이미지당 1ms기다리면서 넘기고 키보드 q 누르면 동영상을 꺼라
#waitKey() <-내부의 숫자가 커질수록 천천히 재생된다. 오래기다렸다가 다음이미지를 보여주니까.
        break