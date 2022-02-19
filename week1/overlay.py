import cv2
#오버레이 이미지는 png파일이여야 하고 배경이 투명인걸 선호해야 한다.//반전이 크게일어나기 때문에
img = cv2.imread('01.jpg')
overlay_img = cv2.imread('dices.png', cv2.IMREAD_UNCHANGED)
#png파일을 사용할때는 투명도를 같이 로드하고 싶어서 imread를 부른다


#3개의채널로는 색깔만 표현할수있고 채널이 하나추가 (A(알파))채널은 투명도를 표현하는 채널이다.
overlay_img = cv2.resize(overlay_img, dsize=(150, 150))

#이미지합성 함수부분
overlay_alpha = overlay_img[:, :, 3:] / 255.0 #알파채널만 발췌하는 코드 255로나눠서 0~1사이의 값으로 만든다
background_alpha = 1.0 - overlay_alpha#주사위의 투명도의 반전된값! 1에서 빼주니까

x1 = 100
y1 = 100
#주사위 이미지를 넣고싶은 왼쪽위 좌표만
x2 = x1 + 150
y2 = y1 + 150
#위의 dsize150 150 이라서 x2,y2도 똑같이 150,150더해준다
img[y1:y2, x1:x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[y1:y2, x1:x2]
#overlay_alpha== 주사위의 투명도값을 주사위 이미지에 곱하여 주사위가 없는 부분은 투명해진다
#overlay_img[:, :, :3]해당부분은 3채널까지만 즉4채널인 알파채널을 제외하고 색깔을 표현해주는 입력층만 갖고오겠다.
#background_alpha==overlay_alpha의 투명도가 반전된값을 곱하게되면 주사위부분만투명하게되고 백그라운드만 불투명하게 바뀐다.
#imgs[y1:y2, x1:x2]여기서는 왜 3채널만 쓰는가? -> 이미지값자체가 원래 3차원(알파채널이 없는 상태)로 입력되었기 때문에


cv2.imshow('imgs',img)
cv2.waitKey(0)
#계속띄워놓을거야