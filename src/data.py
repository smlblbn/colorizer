import numpy as np
import utils

test_np = np.ndarray(shape=(100, 256, 256, 1))

train_np = np.ndarray(shape=(800, 256, 256, 1))
valid_np = np.ndarray(shape=(100, 256, 256, 1))

train_np_gt = np.ndarray(shape=(800, 64, 64, 2))
valid_np_gt = np.ndarray(shape=(100, 64, 64, 2))

train_np_real = np.ndarray(shape=(800, 256, 256, 3))
valid_np_real = np.ndarray(shape=(100, 256, 256, 3))

index = 0
with open('train.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line.strip():
            line = line.split('\n')
            img = utils.read_image('../gray/' + line[0]).astype(np.uint8)
            img, _ = utils.cvt2Lab(img)
            img = img.reshape(256, 256, 1).astype(np.int8)

            img_gt = utils.read_image('../color_64/' + line[0]).astype(np.uint8)
            _, img_gt = utils.cvt2Lab(img_gt)
            img_gt = img_gt.astype(np.int8)

            img_real = utils.read_image('../color_256/' + line[0]).astype(np.uint8)

            train_np[index] = img
            train_np_gt[index] = img_gt
            train_np_real[index] = img_real
            index = index + 1

with open('../train.npy', 'wb') as file:
    np.save(file, train_np)

with open('../train_gt.npy', 'wb') as file:
    np.save(file, train_np_gt)

with open('../train_real.npy', 'wb') as file:
    np.save(file, train_np_real)

index = 0
with open('valid.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line.strip():
            line = line.split('\n')

            img = utils.read_image('../gray/' + line[0]).astype(np.uint8)
            img, _ = utils.cvt2Lab(img)
            img = img.reshape(256, 256, 1).astype(np.int8)

            img_gt = utils.read_image('../color_64/' + line[0]).astype(np.uint8)
            _, img_gt = utils.cvt2Lab(img_gt)
            img_gt = img_gt.astype(np.int8)

            img_real = utils.read_image('../color_256/' + line[0]).astype(np.uint8)

            valid_np[index] = img
            valid_np_gt[index] = img_gt
            valid_np_real[index] = img_real
            index = index + 1

with open('../valid.npy', 'wb') as file:
    np.save(file, valid_np)

with open('../valid_gt.npy', 'wb') as file:
    np.save(file, valid_np_gt)

with open('../valid_real.npy', 'wb') as file:
    np.save(file, valid_np_real)

index = 0
with open('test.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line.strip():
            line = line.split('\n')

            img = utils.read_image('../test_gray/' + line[0]).astype(np.uint8)
            img, _ = utils.cvt2Lab(img)
            img = img.reshape(256, 256, 1).astype(np.int8)

            test_np[index] = img
            index = index + 1

with open('../test.npy', 'wb') as file:
    np.save(file, test_np)