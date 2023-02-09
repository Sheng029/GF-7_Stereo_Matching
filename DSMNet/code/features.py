from modules import *


class FeatureExtractor(keras.Model):
    def __init__(self, filters):
        super(FeatureExtractor, self).__init__()

        self.conv0_1 = conv2d_bn_act(2 * filters, 5, 2, 'same', 1, True)
        self.conv0_2 = conv2d_bn_act(2 * filters, 5, 2, 'same', 1, True)
        self.conv0_3 = make_res_blocks(2 * filters, 1, 6)

        self.conv1_1 = conv2d_bn_act(2 * filters, 3, 1, 'same', 1, True)
        self.conv1_2 = make_res_blocks(2 * filters, 1, 4)
        self.conv1_3 = conv2d(filters, 3, 1, 'same', 1, False)

        self.conv2_1 = conv2d_bn_act(2 * filters, 3, 2, 'same', 1, True)
        self.conv2_2 = make_res_blocks(2 * filters, 1, 4)
        self.conv2_3 = conv2d(filters, 3, 1, 'same', 1, False)

    def call(self, inputs, training=None, mask=None):
        x = self.conv0_1(inputs)
        x = self.conv0_2(x)
        x = self.conv0_3(x)

        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)

        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)
        x2 = self.conv2_3(x2)

        return [x1, x2]     # [high, low]


class HighScaleFeatureExtractor(keras.Model):
    def __init__(self, filters):
        super(HighScaleFeatureExtractor, self).__init__()

        self.conv0_1 = conv2d_bn_act(2 * filters, 5, 2, 'same', 1, True)
        self.conv0_2 = conv2d_bn_act(2 * filters, 5, 2, 'same', 1, True)
        self.conv0_3 = make_res_blocks(2 * filters, 1, 6)

        self.conv1_1 = conv2d_bn_act(2 * filters, 3, 1, 'same', 1, True)
        self.conv1_2 = make_res_blocks(2 * filters, 1, 4)
        self.conv1_3 = conv2d(filters, 3, 1, 'same', 1, False)

    def call(self, inputs, training=None, mask=None):
        x = self.conv0_1(inputs)
        x = self.conv0_2(x)
        x = self.conv0_3(x)

        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)

        return x


class LowScaleFeatureExtractor(keras.Model):
    def __init__(self, filters):
        super(LowScaleFeatureExtractor, self).__init__()

        self.conv0_1 = conv2d_bn_act(2 * filters, 5, 2, 'same', 1, True)
        self.conv0_2 = conv2d_bn_act(2 * filters, 5, 2, 'same', 1, True)
        self.conv0_3 = make_res_blocks(2 * filters, 1, 6)

        self.conv2_1 = conv2d_bn_act(2 * filters, 3, 2, 'same', 1, True)
        self.conv2_2 = make_res_blocks(2 * filters, 1, 4)
        self.conv2_3 = conv2d(filters, 3, 1, 'same', 1, False)

    def call(self, inputs, training=None, mask=None):
        x = self.conv0_1(inputs)
        x = self.conv0_2(x)
        x = self.conv0_3(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        return x
