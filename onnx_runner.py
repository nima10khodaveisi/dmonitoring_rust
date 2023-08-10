import cv2

import numpy as np
import onnxruntime as ort

from camera import find_calib, undist_frame


def sig(x):
    return 1/(1 + np.exp(-x))

ONNX_FILE_NAME = "dmonitoring_model.onnx"

ort_sess = ort.InferenceSession(ONNX_FILE_NAME)
# get image from resized video
# cap = cv2.VideoCapture('interior_center_day.mkv')
cap = cv2.VideoCapture('samples/interior_center-1105.mkv')
# cap = cv2.VideoCapture('resized.mkv')
# cap = cv2.VideoCapture('samples/interior_center-0534.mkv')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('result-python-un.mkv', fourcc, 15, (1440, 960))

cnt = 0

while True:
    ret, frame = cap.read()
    cnt = cnt + 1

    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)
    # exit(0)

    if not ret:
        break


    frame = cv2.resize(frame, (1440, 960))

    dst, _ = undist_frame(frame, (frame.shape[1], frame.shape[0]))

    dst = cv2.resize(dst, (1440, 960))

    height, width = frame.shape[:2]
    print(height, width)

    REG_SCALE = 0.25

    # cv2.imshow('frame', frame)
    calib = find_calib(frame, bool(cnt == 1))
    frame = dst

    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(image)
    image = y.astype('float32') / 255. # dm model just takes the y channel

    # calib = np.array([0, 0, 0], dtype='float32').reshape(1, 3)

    # print("this is calib ", calib)

    input_image = image.reshape(1, -1)
    
    output = ort_sess.run(None, {"input_img": input_image, "calib": calib})[0].reshape(-1).astype('float32')
    
    offset = 0
    
    driver_face_x = int((output[3 + offset] * REG_SCALE + 0.54) * width)
    driver_face_y = int((output[4 + offset] * REG_SCALE + 0.5 - 0.28) * height)


    # driver_face_x = int((output[3 + offset] * 0.4868 + 0.3444) * width)
    # driver_face_y = int((output[4 + offset] * 0.5531 + 0.2865) * height)


    print(output[3], output[4])
    print(driver_face_x, driver_face_y)

    sunglass_prob = sig(output[33])
    touching_wheel_prob = sig(output[35])
    using_phone_prob = sig(output[39])
    distracted_prob = sig(output[40])

    frame_data = [
        "using_phone_prob: " + str(using_phone_prob),
        "sunglass_prob: " + str(sunglass_prob),
        "distracted_prob: " + str(distracted_prob),
        "touching_wheel_prob: " + str(touching_wheel_prob)
    ]

    print(distracted_prob)

    for i in range(len(frame_data)):
        x = 50;
        y = (i + 1) * 50 + 50

        cv2.putText(frame, frame_data[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, lineType = cv2.LINE_AA)

    # put the a circle on the driver's face
    output_image = cv2.circle(frame, (driver_face_x, driver_face_y), radius=70, color=(0, 0, 255), thickness=10)

    output_video.write(output_image)

    # cv2.imshow('frame', output_image)
    # break


output_video.release()

cap.release()
# cv2.waitKey(0)
cv2.destroyAllWindows()



