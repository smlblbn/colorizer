import numpy as np
import utils

l_valid_np = np.load('../valid.npy')
ab_valid_np = np.load('../valid_est.npy')

res_valid_np = np.ndarray(shape=(100, 256, 256, 3))
ab_valid_np = np.transpose(ab_valid_np, (0, 2, 3, 1))

for i in range(100):
    img_lab = np.concatenate((l_valid_np[i].astype(np.uint8), utils.upsample(ab_valid_np[i].astype(np.double))), axis=2)
    img_rgb = utils.cvt2rgb(img_lab)

    img_rgb = img_rgb * 255
    img_rgb = img_rgb.astype(np.uint8)

    res_valid_np[i] = img_rgb

with open('../estimations_valid.npy', 'wb') as file:
    np.save(file, res_valid_np)

l_test_np = np.load('../test.npy')
ab_test_np = np.load('../test_est.npy')

res_test_np = np.ndarray(shape=(100, 256, 256, 3))
ab_test_np = np.transpose(ab_test_np, (0, 2, 3, 1))

for i in range(100):
    img_lab = np.concatenate((l_test_np[i].astype(np.uint8), utils.upsample(ab_test_np[i].astype(np.double))), axis=2)
    img_rgb = utils.cvt2rgb(img_lab)

    img_rgb = img_rgb * 255
    img_rgb = img_rgb.astype(np.uint8)

    res_test_np[i] = img_rgb

with open('../estimations_test.npy', 'wb') as file:
    np.save(file, res_test_np)