'''
采用金字塔匹配策略
'''
import os
from osgeo import gdal
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import cv2
import time


class Predictor:
    def __init__(self, filepath1, filepath2, height, width, model_dir):
        self.image1 = self.readImage(filepath1)
        self.image2 = self.readImage(filepath2)
        self.height = height
        self.width = width
        self.model = keras.models.load_model(filepath=model_dir, compile=False)

    @staticmethod
    def readImage(filepath):
        return gdal.Open(filepath).ReadAsArray()

    @staticmethod
    def writeImage(image, filepath):
        (rows, cols) = image.shape
        driver = gdal.GetDriverByName('GTiff')
        if 'int8' in image.dtype.name:
            dataset = driver.Create(filepath, cols, rows, 1, gdal.GDT_Byte)
        elif 'int16' in image.dtype.name:
            dataset = driver.Create(filepath, cols, rows, 1, gdal.GDT_UInt16)
        else:
            dataset = driver.Create(filepath, cols, rows, 1, gdal.GDT_Float32)
        dataset.GetRasterBand(1).WriteArray(image)
        del dataset

    @staticmethod
    def normImage(image1, image2):
        norm_image1 = (image1 - np.mean(image1)) / np.std(image1)
        norm_image2 = (image2 - np.mean(image2)) / np.std(image2)
        norm_image1 = np.expand_dims(norm_image1, -1)
        norm_image2 = np.expand_dims(norm_image2, -1)
        norm_image1 = np.expand_dims(norm_image1, 0)
        norm_image2 = np.expand_dims(norm_image2, 0)
        return norm_image1, norm_image2

    @staticmethod
    def meanDsp(patch, threshold):
        mask = (patch > threshold).astype('int32')
        patch[patch <= threshold] = 0.0
        return int(np.sum(patch) / np.sum(mask))

    def __predict8x(self):
        height, width = self.height, self.width
        rows, cols = self.image1.shape
        rows, cols = rows // 8, cols // 8
        image1 = cv2.resize(self.image1, (cols, rows))
        image2 = cv2.resize(self.image2, (cols, rows))
        dsp8x = np.zeros((rows, cols), 'float32')
        # 非边缘区
        i, j = 0, 0
        while i + height <= rows:
            while j + width <= cols:
                patch1, patch2 = self.normImage(image1[i:i+height, j:j+width], image2[i:i+height, j:j+width])
                dsp = self.model.predict([patch1, patch2])[0, :, :, 0]
                if i == 0 and j == 0:
                    dsp8x[i:i+height, j:j+width] = dsp
                elif i == 0 and j != 0:
                    dsp8x[i:i+height, j+width//4:j+width] = dsp[:, width//4:width]
                elif i !=0 and j == 0:
                    dsp8x[i+height//4:i+height, j:j+width] = dsp[height//4:height, :]
                else:
                    dsp8x[i+height//4:i+height, j+width//4:j+width] = dsp[height//4:height, width//4:width]
                j += width // 2
            j = 0
            i += height // 2
        # 右边缘
        i = 0
        while i + height <= rows:
            patch1, patch2 = self.normImage(image1[i:i+height, cols-width:cols], image2[i:i+height, cols-width:cols])
            dsp = self.model.predict([patch1, patch2])[0, :, :, 0]
            if i == 0:
                dsp8x[i:i+height, cols-width+width//32:cols] = dsp[:, width//32:width]
            else:
                dsp8x[i+height//4:i+height, cols-width+width//32:cols] = dsp[height//4:height, width//32:width]
            i += height // 2
        # 下边缘
        j = 0
        while j + width <= cols:
            patch1, patch2 = self.normImage(image1[rows-height:rows, j:j+width], image2[rows-height:rows, j:j+width])
            dsp = self.model.predict([patch1, patch2])[0, :, :, 0]
            if j == 0:
                dsp8x[rows-height+height//32:rows, j:j+width] = dsp[height//32:height, :]
            else:
                dsp8x[rows-height+height//32:rows, j+width//4:j+width] = dsp[height//32:height, width//4:width]
            j += width // 2
        # 右下角
        patch1, patch2 = self.normImage(image1[rows-height:rows, cols-width:cols], image2[rows-height:rows, cols-width:cols])
        dsp = self.model.predict([patch1, patch2])[0, :, :, 0]
        dsp8x[rows-height:rows, cols-width:cols] = dsp
        yx = np.where(image2 == 0)
        del image1, image2
        dsp8x[yx] = -32767.0
        return dsp8x

    def __predict(self, scale_factor, pre_dsp, threshold):
        height, width = self.height, self.width
        rows, cols = self.image1.shape
        rows, cols = rows // scale_factor, cols // scale_factor
        image1 = cv2.resize(self.image1, (cols, rows))
        image2 = cv2.resize(self.image2, (cols, rows))
        r = cols / pre_dsp.shape[1]
        pre_dsp = cv2.resize(pre_dsp, (cols, rows)) * r
        dsps = np.zeros((rows, cols), 'float32')
        # 非边缘区
        i, j = 0, 0
        while i + height <= rows:
            while j + width <= cols:
                patch1 = image1[i:i+height, j:j+width]
                patch = pre_dsp[i:i+height, j:j+width]
                avg = self.meanDsp(patch, threshold)
                if i == 0 and j == 0:
                    patch2 = image2[i:i+height, j:j+width]
                    patch1, patch2 = self.normImage(patch1, patch2)
                    dsp = self.model.predict([patch1, patch2])[0, :, :, 0]
                    dsps[i:i+height, j:j+width] = dsp
                elif i == 0 and j != 0:
                    if j - avg > 0 and j - avg + width < cols:
                        patch2 = image2[i:i+height, j-avg:j-avg+width]
                        patch1, patch2 = self.normImage(patch1, patch2)
                        dsp = self.model.predict([patch1, patch2])[0, :, :, 0] + avg
                        dsps[i:i+height, j+width//4:j+width] = dsp[:, width//4:width]
                    else:
                        patch2 = image2[i:i+height, j:j+width]
                        patch1, patch2 = self.normImage(patch1, patch2)
                        dsp = self.model.predict([patch1, patch2])[0, :, :, 0]
                        dsps[i:i+height, j+width//4:j+width] = dsp[:, width//4:width]
                elif i != 0 and j == 0:
                    if j - avg > 0 and j - avg + width < cols:
                        patch2 = image2[i:i+height, j-avg:j-avg+width]
                        patch1, patch2 = self.normImage(patch1, patch2)
                        dsp = self.model.predict([patch1, patch2])[0, :, :, 0] + avg
                        dsps[i+height//4:i+height, j:j+width] = dsp[height//4:height, :]
                    else:
                        patch2 = image2[i:i+height, j:j+width]
                        patch1, patch2 = self.normImage(patch1, patch2)
                        dsp = self.model.predict([patch1, patch2])[0, :, :, 0]
                        dsps[i+height//4:i+height, j:j+width] = dsp[height//4:height, :]
                else:
                    if j - avg > 0 and j - avg + width < cols:
                        patch2 = image2[i:i+height, j-avg:j-avg+width]
                        patch1, patch2 = self.normImage(patch1, patch2)
                        dsp = self.model.predict([patch1, patch2])[0, :, :, 0] + avg
                        dsps[i+height//4:i+height, j+width//4:j+width] = dsp[height//4:height, width//4:width]
                    else:
                        patch2 = image2[i:i+height, j:j+width]
                        patch1, patch2 = self.normImage(patch1, patch2)
                        dsp = self.model.predict([patch1, patch2])[0, :, :, 0]
                        dsps[i+height//4:i+height, j+width//4:j+width] = dsp[height//4:height, width//4:width]
                j += width // 2
            j = 0
            i += height // 2
        # 右边缘
        i = 0
        j = cols - width
        while i + height <= rows:
            patch1 = image1[i:i+height, j:j+width]
            patch = pre_dsp[i:i+height, j:j+width]
            avg = self.meanDsp(patch, threshold)
            if i == 0:
                if j - avg > 0 and j - avg + width < cols:
                    patch2 = image2[i:i+height, j-avg:j-avg+width]
                    patch1, patch2 = self.normImage(patch1, patch2)
                    dsp = self.model.predict([patch1, patch2])[0, :, :, 0] + avg
                    dsps[i:i+height, cols-width+width//32:cols] = dsp[:, width//32:width]
                else:
                    patch2 = image2[i:i+height, j:j+width]
                    patch1, patch2 = self.normImage(patch1, patch2)
                    dsp = self.model.predict([patch1, patch2])[0, :, :, 0]
                    dsps[i:i + height, cols - width + width // 32:cols] = dsp[:, width // 32:width]
            else:
                if j - avg > 0 and j - avg + width < cols:
                    patch2 = image2[i:i+height, j-avg:j-avg+width]
                    patch1, patch2 = self.normImage(patch1, patch2)
                    dsp = self.model.predict([patch1, patch2])[0, :, :, 0] + avg
                    dsps[i+height//4:i+height, cols-width+width//32:cols] = dsp[height//4:height, width//32:width]
                else:
                    patch2 = image2[i:i+height, j:j+width]
                    patch1, patch2 = self.normImage(patch1, patch2)
                    dsp = self.model.predict([patch1, patch2])[0, :, :, 0]
                    dsps[i+height//4:i+height, cols-width+width//32:cols] = dsp[height//4:height, width//32:width]
            i += height // 2
        # 下边缘
        j = 0
        while j + width <= cols:
            patch1 = image1[rows-height:rows, j:j+width]
            patch = pre_dsp[rows-height:rows, j:j+width]
            avg = self.meanDsp(patch, threshold)
            if j == 0:
                if j - avg > 0 and j - avg + width < cols:
                    patch2 = image2[rows-height:rows, j-avg:j-avg+width]
                    patch1, patch2 = self.normImage(patch1, patch2)
                    dsp = self.model.predict([patch1, patch2])[0, :, :, 0] + avg
                    dsps[rows-height+height//32:rows, j:j+width] = dsp[height//32:height, :]
                else:
                    patch2 = image2[rows-height:rows, j:j+width]
                    patch1, patch2 = self.normImage(patch1, patch2)
                    dsp = self.model.predict([patch1, patch2])[0, :, :, 0]
                    dsps[rows-height+height//32:rows, j:j+width] = dsp[height//32:height, :]
            else:
                if j - avg > 0 and j - avg + width < cols:
                    patch2 = image2[rows-height:rows, j-avg:j-avg+width]
                    patch1, patch2 = self.normImage(patch1, patch2)
                    dsp = self.model.predict([patch1, patch2])[0, :, :, 0] + avg
                    dsps[rows-height+height//32:rows, j+width//4:j+width] = dsp[height//32:height, width//4:width]
                else:
                    patch2 = image2[rows-height:rows, j:j+width]
                    patch1, patch2 = self.normImage(patch1, patch2)
                    dsp = self.model.predict([patch1, patch2])[0, :, :, 0]
                    dsps[rows - height + height // 32:rows, j + width // 4:j + width] = dsp[height // 32:height, width // 4:width]
            j += width // 2
        # 右下角
        j = cols - width
        patch1 = image1[rows-height:rows, j:j+width]
        patch = pre_dsp[rows-height:rows, j:j+width]
        avg = self.meanDsp(patch, threshold)
        if j - avg > 0 and j - avg + width < cols:
            patch2 = image2[rows - height:rows, j - avg:j - avg + width]
            patch1, patch2 = self.normImage(patch1, patch2)
            dsp = self.model.predict([patch1, patch2])[0, :, :, 0] + avg
            dsps[rows-height:rows, cols-width:cols] = dsp
        else:
            patch2 = image2[rows - height:rows, j:j + width]
            patch1, patch2 = self.normImage(patch1, patch2)
            dsp = self.model.predict([patch1, patch2])[0, :, :, 0]
            dsps[rows - height:rows, cols - width:cols] = dsp
        yx = np.where(image2 == 0)
        del image1, image2, pre_dsp
        dsps[yx] = -32767.0
        return dsps

    def predictDisparity(self, save_dir, filepath):
        t1 = time.time()
        dsp8x = self.__predict8x()
        self.writeImage(dsp8x, os.path.join(save_dir, 'x8.tif'))
        dsp4x = self.__predict(4, dsp8x, -512.0)
        del dsp8x
        self.writeImage(dsp4x, os.path.join(save_dir, 'x4.tif'))
        dsp2x = self.__predict(2, dsp4x, -1024.0)
        del dsp4x
        self.writeImage(dsp2x, os.path.join(save_dir, 'x2.tif'))
        dsp1x = self.__predict(1, dsp2x, -2048.0)
        del dsp2x
        self.writeImage(dsp1x, filepath)
        del dsp1x
        t2 = time.time()
        print('Time: %.6f' % (t2 - t1))


def main_func(memory, filepath1, filepath2, height, width, model_dir, save_dir, filepath):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * memory)])

    predictor = Predictor(filepath1, filepath2, height, width, model_dir)
    predictor.predictDisparity(save_dir, filepath)


if __name__ == '__main__':
    memory = 24
    filepath1 = '/home/hesheng/Test/Jinan/GF7_DLC_E117.1_N36.6_20210202_L1A0000323606-BWDPAN.tiff'
    filepath2 = '/home/hesheng/Test/Jinan/GF7_DLC_E117.1_N36.6_20210202_L1A0000323606-FWDPAN.tiff'
    height, width = 2048, 2048
    model_dir = '/home/hesheng/Networks/DSMNet/model/DSMNetMIX2048'
    save_dir = '/home/hesheng/Test/Jinan'
    filepath = '/home/hesheng/Test/Jinan/GF7_DLC_E117.1_N36.6_20210202_L1A0000323606-BWDPAN_BF_LR_disp_1.tif'
    main_func(memory, filepath1, filepath2, height, width, model_dir, save_dir, filepath)
