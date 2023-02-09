import tensorflow as tf
import tensorflow.keras as keras
from aggregations import conv3d


class DisparityComputation(keras.Model):
    def __init__(self, min_disp, max_disp):
        super(DisparityComputation, self).__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.conv = conv3d(1, (1, 1, 1), (1, 1, 1), 'valid', False)

    def call(self, inputs, training=None, mask=None):
        cost_volume = self.conv(inputs)     # [N, D, H, W, 1]
        cost_volume = tf.squeeze(cost_volume, -1)     # [N, D, H, W]
        cost_volume = tf.transpose(cost_volume, (0, 2, 3, 1))   # [N, H, W, D]
        assert cost_volume.shape[-1] == self.max_disp - self.min_disp
        candidates = tf.linspace(1.0 * self.min_disp, 1.0 * self.max_disp - 1.0, self.max_disp - self.min_disp)
        probabilities = tf.math.softmax(-1.0 * cost_volume, -1)
        disparity = tf.reduce_sum(candidates * probabilities, -1, True)

        return disparity     # [N, H, W, 1]
