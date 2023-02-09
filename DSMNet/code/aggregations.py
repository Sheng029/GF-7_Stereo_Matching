import tensorflow as tf
import tensorflow.keras as keras
from modules import L2


def conv3d(filters, kernel_size, strides, padding, use_bias=False):
    return keras.layers.Conv3D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
        activation=None, use_bias=use_bias, kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(L2)
    )


def conv3d_bn_act(filters, kernel_size, strides, padding, activation=True):
    conv = keras.layers.Conv3D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
        activation=None, use_bias=False, kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(L2)
    )
    bn = keras.layers.BatchNormalization()
    leaky_relu = keras.layers.LeakyReLU()

    if activation:
        return keras.Sequential([conv, bn, leaky_relu])
    else:
        return keras.Sequential([conv, bn])


def trans_conv3d_bn_act(filters, kernel_size, strides, padding, activation=True):
    conv = keras.layers.Conv3DTranspose(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
        activation=None, use_bias=False, kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(L2)
    )
    bn = keras.layers.BatchNormalization()
    leaky_relu = keras.layers.LeakyReLU()

    if activation:
        return keras.Sequential([conv, bn, leaky_relu])
    else:
        return keras.Sequential([conv, bn])


def factorize_conv3d(filters):
    conv1 = conv3d(filters, (3, 1, 1), (1, 1, 1), 'same', False)
    conv2 = conv3d(filters, (1, 3, 3), (1, 1, 1), 'same', False)
    bn = keras.layers.BatchNormalization()
    leaky_relu = keras.layers.LeakyReLU()

    return keras.Sequential([conv1, conv2, bn, leaky_relu])


class FactorizedCostAggregation(keras.Model):
    def __init__(self, filters):
        super(FactorizedCostAggregation, self).__init__()

        self.conv1 = conv3d_bn_act(filters, (3, 3, 3), (1, 1, 1), 'same', True)
        self.conv2 = factorize_conv3d(filters)
        self.conv3 = factorize_conv3d(filters)
        self.conv4 = conv3d_bn_act(2 * filters, (3, 3, 3), (2, 2, 2), 'same', True)
        self.conv5 = factorize_conv3d(2 * filters)
        self.conv6 = factorize_conv3d(2 * filters)
        self.conv7 = conv3d_bn_act(4 * filters, (3, 3, 3), (2, 2, 2), 'same', True)
        self.conv8 = factorize_conv3d(4 * filters)
        self.conv9 = factorize_conv3d(4 * filters)
        self.conv10 = trans_conv3d_bn_act(2 * filters, (3, 3, 3), (2, 2, 2), 'same', True)
        self.conv11 = factorize_conv3d(2 * filters)
        self.conv12 = trans_conv3d_bn_act(filters, (3, 3, 3), (2, 2, 2), 'same', True)
        self.conv13 = factorize_conv3d(filters)
        self.conv14 = conv3d(filters, (3, 3, 3), (1, 1, 1), 'same', False)

    def call(self, inputs, training=None, mask=None):
        x1 = self.conv1(inputs)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x2 = self.conv4(x1)
        x2 = self.conv5(x2)
        x2 = self.conv6(x2)
        x3 = self.conv7(x2)
        x3 = self.conv8(x3)
        x3 = self.conv9(x3)
        x4 = self.conv10(x3)
        x4 += x2
        x4 = self.conv11(x4)
        x5 = self.conv12(x4)
        x5 += x1
        x5 = self.conv13(x5)
        x = self.conv14(x5)    # [N, D, H, W, C]

        return x


class PlainAggregation(keras.Model):
    def __init__(self, filters):
        super(PlainAggregation, self).__init__()

        self.conv1 = conv3d_bn_act(filters, (3, 3, 3), (1, 1, 1), 'same', True)
        self.conv2 = conv3d_bn_act(filters, (3, 3, 3), (1, 1, 1), 'same', True)
        self.conv3 = conv3d_bn_act(2 * filters, (3, 3, 3), (1, 1, 1), 'same', True)
        self.conv4 = conv3d_bn_act(2 * filters, (3, 3, 3), (1, 1, 1), 'same', True)
        self.conv5 = conv3d_bn_act(4 * filters, (3, 3, 3), (1, 1, 1), 'same', True)
        self.conv6 = conv3d_bn_act(4 * filters, (3, 3, 3), (1, 1, 1), 'same', True)
        self.conv7 = conv3d_bn_act(2 * filters, (3, 3, 3), (1, 1, 1), 'same', True)
        self.conv8 = conv3d(filters, (3, 3, 3), (1, 1, 1), 'same', False)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        return x
