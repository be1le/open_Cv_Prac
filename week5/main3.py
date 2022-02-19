import cv2
#모데들들 임포트 해오기.
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel('models/EDSR_x3.pb')
sr.setModel('edsr', 3)
#모델의 output값이 input값의 3배이다.


img = cv2.imread('imgs/06.jpg')
#img변수에 로드한 이미지 할당

result = sr.upsampe(img)
#result에 upsampe(img)할당

resized_img = cv2.resize(img, dsize=None, fx=3, fy=3)
#비교를위해 원래 input 이미지도 3배를 늘리는 구문.


#imshow로 이미지 출력하는 구문. 
cv2.imshow('img', img)
cv2.imshow('resized_img', resized_img)
cv2.imshow('result', result)
cv2.waitKey(0)