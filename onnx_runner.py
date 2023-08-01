import cv2

import numpy as np
import onnxruntime as ort


ONNX_FILE_NAME = "dmonitoring_model.onnx"

ort_sess = ort.InferenceSession(ONNX_FILE_NAME)
# get image from resized video
cap = cv2.VideoCapture('interior_center_day.mkv')
# cap = cv2.VideoCapture('resized.mkv')
cnt = 0
while True:
    ret, frame = cap.read()
    cnt = cnt + 1

    if cnt < 100:
        continue 
    frame = cv2.resize(frame, (1440, 960))
    height, width = frame.shape[:2]
    print(height, width)
    W = 1928
    H = 1208

    REG_SCALE = 0.25

    # cv2.imshow('frame', frame)

    frame = frame.astype('float32') / 255.
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(image)
    image = y # dm model just takes the y channel

    calib = np.array([-10000, -10000, -100000], dtype='float32').reshape(1, 3)

    input_image = image.reshape(1, -1)
    
    output = ort_sess.run(None, {"input_img": input_image, "calib": calib})[0].reshape(-1).astype('float32')
    
    offset = 0

    driver_face_x = int((output[3 + offset] * REG_SCALE + 0.5) * width)
    driver_face_y = int((output[4 + offset] * REG_SCALE + 0.5) * height)


    print(output[3] * 0.25, output[4] * 0.25)
    print(driver_face_x, driver_face_y)

    # put the a circle on the driver's face
    output_image = cv2.circle(frame, (driver_face_x, driver_face_y), radius=70, color=(0, 0, 255), thickness=10)

    cv2.imshow('frame', output_image)

    break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()



