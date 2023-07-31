import onnx
import cv2

import numpy as np
import onnxruntime as ort


ONNX_FILE_NAME = "dmonitoring_model.onnx"

# image = np.random.random_sample((1440, 960)).astype('float32').reshape(1, -1)
ort_sess = ort.InferenceSession(ONNX_FILE_NAME)
# get image from resized video
cap = cv2.VideoCapture('resized.mkv')
cnt = 0
while True:
    ret, frame = cap.read()
    cnt = cnt + 1

    if cnt < 210:
        continue 

    height, width = frame.shape[:2]

    frame = cv2.resize(frame, (1440, 960))
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(image)
    image = y # dm model just takes y channel

    calib = np.array([80, 40, 72], dtype='float32').reshape(1, 3)

    input_image = image.reshape(1, -1).astype('float32') / 255.
    
    output = ort_sess.run(None, {"input_img": input_image, "calib": calib})[0].reshape(-1)
    
    driver_face_x = int((output[3] + 0.5) * width)
    driver_face_y = int((output[4] + 0.5) * height)


    print(driver_face_x, driver_face_y, frame.shape)
    output_image = cv2.circle(cv2.resize(frame, (1928, 1208)), (int(driver_face_x), int(driver_face_y)), radius=50, color=(0, 0, 255), thickness=10)

    cv2.imshow('frame', output_image)

    break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()



