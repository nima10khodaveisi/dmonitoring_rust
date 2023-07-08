import onnx

import numpy as np
import onnxruntime as ort


ONNX_FILE_NAME = "dmonitoring_model.onnx"

image = np.random.random_sample((1440, 960)).astype('float32').reshape(1, -1)

calib = np.array([0, 0, 0], dtype='float32').reshape(1, 3)

ort_sess = ort.InferenceSession(ONNX_FILE_NAME)

output = ort_sess.run(None, { "input_img": image, "calib": calib})

print(output[0])
print(output[0].shape)
