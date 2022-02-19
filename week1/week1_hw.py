import cv2
#cv2를 로드해오기
cap = cv2.VideoCapture(0)
#VideoCapture는 동영상 로드할때 사용하는 메서드.
while True:
    ret, img = cap.read()
    # cap.read()가 img라는 변수에 img를 한장씩 저장해주는 역할을 한다.
    if not ret:
        break
    # ret == Flase의 의미는 동영상이 끝나면 루프를 빠져나와라!
    cropped_img = img[183:465, 721:878]
    #이미지를 cropped할때는 y,x순으로 적어주기.
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    #cv2.COLOR_BGR2GRAY해당 구문으로 흑백으로 출력하기

    cv2.imshow('cropped_img', cropped_img)
    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break