import tensorflow as tf
from tensorflow.keras import losses


class SmoothL1Loss(losses.Loss):
    def __init__(self, min_disp, max_disp):
        super(SmoothL1Loss, self).__init__()
        self.min_disp = min_disp
        self.max_disp = max_disp

    def call(self, y_true, y_pred):
        ones = tf.ones_like(y_true, tf.int32)
        zeros = tf.zeros_like(y_true, tf.int32)
        mask1 = tf.where(y_true >= self.max_disp, zeros, ones)
        mask2 = tf.where(y_true < self.min_disp, zeros, ones)
        mask = tf.cast(mask1 & mask2, tf.float32)

        diff = tf.abs(y_true - y_pred)
        less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
        loss = less_than_one * 0.5 * diff ** 2 + (1.0 - less_than_one) * (diff - 0.5)
        return tf.reduce_sum(mask * loss) / tf.reduce_sum(mask)
