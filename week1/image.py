import cv2

img = cv2.imread('01.jpg')

print(img)
print(img.shape)#/높이,너비,채널(색깔정보를포함)보통 rgb(3채널)로이루어짐

cv2.rectangle(img, pt1=(259, 89), pt2=(380, 348), color=(255, 0, 0), thickness=2)
#사각형은 두개의 점만 있어도 만들 수 있다. thickness는 굵기.

cv2.circle(img, center=(320, 220), radius=100, color=(0, 0, 255),thickness=3)
#원은 원의 중심과 반지름의 크기를 정해주면 된다.

cropped_img = img[89:348, 259:380]
#이미지를 자를때는 y,x축 순서로 알려주어야 한다.

img_resized = cv2.resize(img,(512, 256))
#이미지를 resizing할때는 x,y순으로 적용

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#COLOR_BGR2GRAY == 흑백으로 바꾸기

cv2.imshow('result', img_rgb)
cv2.imshow('imgs',img)
cv2.imshow('resized', img_resized)
cv2.imshow('crop', cropped_img)
cv2.waitKey(0)#키보드누를때까지 기다려