import tensorflow.keras as keras

L2 = 1.0e-5


def conv2d(filters, kernel_size, strides, padding, dilation_rate, use_bias=False):
    return keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
        dilation_rate=dilation_rate, activation=None, use_bias=use_bias,
        kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(L2)
    )


def conv2d_bn_act(filters, kernel_size, strides, padding, dilation_rate, activation=True):
    conv = keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
        dilation_rate=dilation_rate, activation=None, use_bias=False,
        kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(L2)
    )
    bn = keras.layers.BatchNormalization()
    relu = keras.layers.ReLU()

    if activation:
        return keras.Sequential([conv, bn, relu])
    else:
        return keras.Sequential([conv, bn])


class ResBlock(keras.Model):
    def __init__(self, filters, dilation_rate):
        super(ResBlock, self).__init__()

        self.conv1 = conv2d_bn_act(filters, 3, 1, 'same', dilation_rate, True)
        self.conv2 = conv2d_bn_act(filters, 3, 1, 'same', dilation_rate, False)
        self.relu = keras.layers.ReLU()

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x += inputs
        x = self.relu(x)

        return x


def make_res_blocks(filters, dilation_rate, num):
    blocks = keras.Sequential()
    for i in range(num):
        blocks.add(ResBlock(filters, dilation_rate))

    return blocks
