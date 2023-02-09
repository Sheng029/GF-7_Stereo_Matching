import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
from feature import FeatureExtractor
from cost import Difference
from aggregation import CostAggregation
from refinement import Refinement
from computation import Computation
from data_reader import read_image, load_batch
from loss_function import SmoothL1Loss
from schedule import schedule


class StereoNet:
    def __init__(self, height, width, channel, min_disp, max_disp):
        self.height = height
        self.width = width
        self.channel = channel
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.model = None

    def build_model(self):
        left_image = keras.Input(shape=(self.height, self.width, self.channel))
        right_image = keras.Input(shape=(self.height, self.width, self.channel))

        extractor = FeatureExtractor(filters=32)
        left_feature = extractor(left_image)
        right_feature = extractor(right_image)

        constructor = Difference(self.min_disp // 8, self.max_disp // 8)
        cost_volume = constructor([left_feature, right_feature])

        aggregator = CostAggregation(filters=32)
        cost_volume = aggregator(cost_volume)

        computer = Computation(self.min_disp // 8, self.max_disp // 8)
        d0 = computer(cost_volume)

        refiner1 = Refinement(filters=32)
        left_image_4x = tf.image.resize(left_image, [self.height // 4, self.width // 4])
        d1 = refiner1([d0, left_image_4x])

        refiner2 = Refinement(filters=32)
        left_image_2x = tf.image.resize(left_image, [self.height // 2, self.width // 2])
        d2 = refiner2([d1, left_image_2x])

        refiner3 = Refinement(filters=32)
        d3 = refiner3([d2, left_image])

        d0 = tf.image.resize(d0, [self.height, self.width]) * 8
        d1 = tf.image.resize(d1, [self.height, self.width]) * 4
        d2 = tf.image.resize(d2, [self.height, self.width]) * 2

        self.model = keras.Model(inputs=[left_image, right_image], outputs=[d0, d1, d2, d3])
        self.model.summary()

    def build_single_output_model(self):
        left_image = keras.Input(shape=(self.height, self.width, self.channel))
        right_image = keras.Input(shape=(self.height, self.width, self.channel))

        extractor = FeatureExtractor(filters=32)
        left_feature = extractor(left_image)
        right_feature = extractor(right_image)

        constructor = Difference(self.min_disp // 8, self.max_disp // 8)
        cost_volume = constructor([left_feature, right_feature])

        aggregator = CostAggregation(filters=32)
        cost_volume = aggregator(cost_volume)

        computer = Computation(self.min_disp // 8, self.max_disp // 8)
        d0 = computer(cost_volume)

        refiner1 = Refinement(filters=32)
        left_image_4x = tf.image.resize(left_image, [self.height // 4, self.width // 4])
        d1 = refiner1([d0, left_image_4x])

        refiner2 = Refinement(filters=32)
        left_image_2x = tf.image.resize(left_image, [self.height // 2, self.width // 2])
        d2 = refiner2([d1, left_image_2x])

        refiner3 = Refinement(filters=32)
        d3 = refiner3([d2, left_image])

        d0 = tf.image.resize(d0, [self.height, self.width]) * 8
        d1 = tf.image.resize(d1, [self.height, self.width]) * 4
        d2 = tf.image.resize(d2, [self.height, self.width]) * 2

        self.model = keras.Model(inputs=[left_image, right_image], outputs=d3)
        self.model.summary()

    def train_only(self, data_dir, weights_save_path, epochs, batch_size, pre_trained_weights):
        # all paths
        all_left_paths = glob.glob(data_dir + '/left/*')
        all_right_paths = glob.glob(data_dir + '/right/*')
        all_disp_paths = glob.glob(data_dir + '/disp/*')

        # sort, necessary in Linux
        all_left_paths.sort()
        all_right_paths.sort()
        all_disp_paths.sort()

        # callbacks
        lr = keras.callbacks.LearningRateScheduler(schedule=schedule, verbose=1)
        mc = keras.callbacks.ModelCheckpoint(filepath=weights_save_path, monitor='refinement_2_loss',
                                             verbose=1, save_best_only=True, save_weights_only=True,
                                             mode='min', save_freq='epoch')

        optimizer = keras.optimizers.Adam()
        loss = [SmoothL1Loss(1.0 * self.min_disp, 1.0 * self.max_disp),
                SmoothL1Loss(1.0 * self.min_disp, 1.0 * self.max_disp),
                SmoothL1Loss(1.0 * self.min_disp, 1.0 * self.max_disp),
                SmoothL1Loss(1.0 * self.min_disp, 1.0 * self.max_disp)]

        if pre_trained_weights is not None:
            self.model.load_weights(filepath=pre_trained_weights, by_name=True)
        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.fit_generator(
            generator=load_batch(all_left_paths, all_right_paths, all_disp_paths, batch_size, True),
            steps_per_epoch=len(all_disp_paths) // batch_size, epochs=epochs, callbacks=[lr, mc],
            shuffle=False)


def predict(left_dir, right_dir, output_dir, model_dir):
    model = keras.models.load_model(model_dir, compile=False)
    lefts = os.listdir(left_dir)
    rights = os.listdir(right_dir)
    lefts.sort()
    rights.sort()
    assert len(lefts) == len(rights)
    for left, right in zip(lefts, rights):
        left_image = read_image(os.path.join(left_dir, left))
        right_image = read_image(os.path.join(right_dir, right))
        left_image = np.expand_dims(left_image, 0)
        right_image = np.expand_dims(right_image, 0)
        disparity = model.predict([left_image, right_image])
        disparity = disparity[0, :, :, 0]
        disparity = Image.fromarray(disparity)
        name = left.replace('left', 'disparity')
        disparity.save(os.path.join(output_dir, name))
