import tensorflow as tf
import tensorflow.keras as keras
from modules import conv2d, L2


def conv2d_bn_leaky(filters, kernel_size, strides, padding, dilation_rate):
    conv = keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
        dilation_rate=dilation_rate, activation=None, use_bias=False,
        kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(L2))
    bn = keras.layers.BatchNormalization()
    leaky_relu = keras.layers.LeakyReLU()

    return keras.Sequential([conv, bn, leaky_relu])


class Refinement(keras.Model):
    def __init__(self, filters):
        super(Refinement, self).__init__()

        self.conv0_1 = conv2d_bn_leaky(filters, 5, 2, 'same', 1)
        self.conv0_2 = conv2d_bn_leaky(filters, 3, 1, 'same', 1)
        self.conv0_3 = conv2d(3, 1, 1, 'valid', 1, False)

        self.conv1_1 = conv2d_bn_leaky(filters, 3, 1, 'same', 1)
        self.conv1_2 = conv2d_bn_leaky(filters, 3, 1, 'same', 1)
        self.conv1_3 = conv2d_bn_leaky(filters, 3, 1, 'same', 2)
        self.conv1_4 = conv2d_bn_leaky(filters, 3, 1, 'same', 4)
        self.conv1_5 = conv2d_bn_leaky(filters, 3, 1, 'same', 1)
        self.conv1_6 = conv2d(1, 3, 1, 'same', 1, False)

    def call(self, inputs, training=None, mask=None):
        # inputs: [disparity, left image], [(N, H, W, 1), ([N, H, W, 3)])
        assert len(inputs) == 2

        # extract shallow features from left image.
        features = self.conv0_1(inputs[1])
        features = self.conv0_2(features)
        features = self.conv0_3(features)

        # Up-sample disparity
        scale_factor = features.shape[1] / inputs[0].shape[1]
        disparity = tf.image.resize(images=inputs[0], size=[features.shape[1], features.shape[2]])
        disparity *= scale_factor

        # learning disparity residual
        con = tf.concat([disparity, features], -1)
        residual = self.conv1_1(con)
        residual = self.conv1_2(residual)
        residual = self.conv1_3(residual)
        residual = self.conv1_4(residual)
        residual = self.conv1_5(residual)
        residual = self.conv1_6(residual)

        refined_disparity = disparity + residual

        return refined_disparity
