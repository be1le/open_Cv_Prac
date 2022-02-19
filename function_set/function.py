# """
# 1. readNet()
# readNet() 함수는 전달된 framework 문자열, 또는 model과 config 파일 이름 확장자를 분석하여 내부에서 해당 프레임워크에 맞는 readNetFromXXX() 형태의 함수를 다시 호출해준다.
#
# 모델 확장자에 따라 호출하는 함수가 다르다.
# .caffemodel -> readNetFromCaffe()
# .pb -> readNetFromTensorflow()
# readNetFromTorch(), readNetFromDarknet(), readNetFromModelOptimizer(), readNetFromONNX()
#
# model_path = './opencv_face_detector_uint8.pb'
# config_path = './opencv_face_detector.pbtxt'
#
# Net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
# 하지만 OpenCV 4.0.0 부터는 확장자 상관없이 readNet()을 사용하는 것이 좋다.
#
# bool Net::emtpy() const;
# 	# 네트워크가 비어있다면 True 를 준다.
#
# # python에서 이런식으로 쓸 수도 있다.
# if net.emtpy():
# 	print('Not Network')
#     sys.exit()
#
#
# #2. blobFromImage()
# Net에 입력되는 데이터는 blob 형식으로 변경 해줘야 한다.
# blob: opencv에서 Mat타입의 4차원(4D Tensor: NCHW) 행렬
# N: 영상 개수
# C: 채널 개수
# H: 영상 세로
# W: 영상 가로
#
# blobFromImages(): 이미지가 두개 이상인 경우 사용
#
# blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300, 300)),
# 	1.0, (300, 300), (104.0, 177.0, 123.0))
#
# 3. setInput()
# Net에 blob 형태의 데이터 넣어주는 함수
#
# Net.setInput(blob)
#
#
# 4. forward()
# Net을 실행 시켜줌(순방향)
#
# prob = Net.forward()
#
# """