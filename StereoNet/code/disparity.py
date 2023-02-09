import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from osgeo import gdal
import time


def readTif(filepath):
    data = gdal.Open(filepath).ReadAsArray()
    return data


def writeTif(data, filepath):
    (rows, cols) = data.shape
    driver = gdal.GetDriverByName('GTiff')
    if 'int8' in data.dtype.name:
        dataset = driver.Create(filepath, cols, rows, 1, gdal.GDT_Byte)
    elif 'int16' in data.dtype.name:
        dataset = driver.Create(filepath, cols, rows, 1, gdal.GDT_UInt16)
    else:
        dataset = driver.Create(filepath, cols, rows, 1, gdal.GDT_Float32)
    dataset.GetRasterBand(1).WriteArray(data)
    del dataset


def normImage(left_patch, right_patch):
    left_patch = (left_patch - np.mean(left_patch)) / np.std(left_patch)
    right_patch = (right_patch - np.mean(right_patch)) / np.std(right_patch)
    left_patch = np.expand_dims(left_patch, -1)
    left_patch = np.expand_dims(left_patch, 0)
    right_patch = np.expand_dims(right_patch, -1)
    right_patch = np.expand_dims(right_patch, 0)
    return left_patch, right_patch


def genDisparityMap(left_img_path, right_img_path, height, width, model_dir):
    # 加载模型
    model = keras.models.load_model(filepath=model_dir, compile=False)
    t1 = time.time()

    left_img = readTif(left_img_path)
    right_img = readTif(right_img_path)
    assert left_img.shape == right_img.shape
    (rows, cols) = left_img.shape
    disparity = np.zeros((rows, cols), 'float32')

    # 非边缘
    i, j = 0, 0
    while i + height <= rows:
        while j + width <= cols:
            left_patch = left_img[i:i+height, j:j+width]
            right_patch = right_img[i:i+height, j:j+width]
            left_patch, right_patch = normImage(left_patch, right_patch)
            disp = model.predict([left_patch, right_patch])[0, :, :, 0]
            if i == 0 and j == 0:
                disparity[i:i+height, j:j+width] = disp
            elif i == 0 and j != 0:
                disparity[i:i+height, j+width//4:j+width] = disp[:, width//4:width]
            elif i != 0 and j == 0:
                disparity[i+height//4:i+height, j:j+width] = disp[height//4:height, :]
            else:
                disparity[i+height//4:i+height, j+width//4:j+width] = disp[height//4:height, width//4:width]
            j += width//2
        j = 0
        i += height//2

    # 右边缘
    i = 0
    while i + height <= rows:
        left_patch = left_img[i:i+height, cols-width:cols]
        right_patch = right_img[i:i+height, cols-width:cols]
        left_patch, right_patch = normImage(left_patch, right_patch)
        disp = model.predict([left_patch, right_patch])[0, :, :, 0]
        if i == 0:
            disparity[i:i + height, cols - width + width // 32:cols] = disp[:, width // 32:width]
        else:
            disparity[i+height//4:i+height, cols - width + width // 32:cols] = disp[height//4:height, width // 32:width]
        i += height//2

    # 下边缘
    j = 0
    while j + width <= cols:
        left_patch = left_img[rows-height:rows, j:j+width]
        right_patch = right_img[rows-height:rows, j:j+width]
        left_patch, right_patch = normImage(left_patch, right_patch)
        disp = model.predict([left_patch, right_patch])[0, :, :, 0]
        if j == 0:
            disparity[rows - height + height // 32:rows, j:j + width] = disp[height // 32:height, :]
        else:
            disparity[rows - height + height // 32:rows, j+width//4:j+width] = disp[height//32:height, width//4:width]
        j += width//2

    # 右下角
    left_patch = left_img[rows-height:rows, cols-width:cols]
    right_patch = right_img[rows-height:rows, cols-width:cols]
    left_patch, right_patch = normImage(left_patch, right_patch)
    disp = model.predict([left_patch, right_patch])[0, :, :, 0]
    disparity[rows-height:rows, cols-width:cols] = disp

    yx = np.where(right_img == 0)
    del left_img
    del right_img
    disparity[yx] = -32767.0

    t2 = time.time()
    print('Total time: %.6f' % (t2 - t1))

    return disparity
