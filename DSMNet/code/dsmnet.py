import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
from features import FeatureExtractor
from costs import CostDifference
from aggregations import FactorizedCostAggregation
from computation import DisparityComputation
from refinement import Refinement
from data_reader import load_batch, read_image
from scheduler import schedule
from loss_function import SmoothL1Loss


class DSMNet:
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

        feature_extractor = FeatureExtractor(filters=16)
        [left_high_feature, left_low_feature] = feature_extractor(left_image)
        [right_high_feature, right_low_feature] = feature_extractor(right_image)

        high_cost_difference = CostDifference(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        low_cost_difference = CostDifference(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        high_cost_volume = high_cost_difference([left_high_feature, right_high_feature])
        low_cost_volume = low_cost_difference([left_low_feature, right_low_feature])

        low_aggregation = FactorizedCostAggregation(filters=16)
        low_agg_cost_volume = low_aggregation(low_cost_volume)

        low_computation = DisparityComputation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        low_disparity = low_computation(low_agg_cost_volume)  # 1/8

        upsample = keras.layers.UpSampling3D(size=(2, 2, 2))
        low_to_high = upsample(low_agg_cost_volume)
        high_cost_volume += low_to_high

        high_aggregation = FactorizedCostAggregation(filters=16)
        high_agg_cost_volume = high_aggregation(high_cost_volume)

        high_computation = DisparityComputation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        high_disparity = high_computation(high_agg_cost_volume)  # 1/4

        refine = Refinement(filters=16)
        refined_disparity = refine([high_disparity, left_image])

        d0 = tf.image.resize(low_disparity, [self.height, self.width]) * 8
        d1 = tf.image.resize(high_disparity, [self.height, self.width]) * 4
        d2 = tf.image.resize(refined_disparity, [self.height, self.width]) * 2

        self.model = keras.Model(inputs=[left_image, right_image], outputs=[d0, d1, d2])
        self.model.summary()

    def build_single_output_model(self):
        left_image = keras.Input(shape=(self.height, self.width, self.channel))
        right_image = keras.Input(shape=(self.height, self.width, self.channel))

        feature_extractor = FeatureExtractor(filters=16)
        [left_high_feature, left_low_feature] = feature_extractor(left_image)
        [right_high_feature, right_low_feature] = feature_extractor(right_image)

        high_cost_difference = CostDifference(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        low_cost_difference = CostDifference(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        high_cost_volume = high_cost_difference([left_high_feature, right_high_feature])
        low_cost_volume = low_cost_difference([left_low_feature, right_low_feature])

        low_aggregation = FactorizedCostAggregation(filters=16)
        low_agg_cost_volume = low_aggregation(low_cost_volume)

        low_computation = DisparityComputation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        low_disparity = low_computation(low_agg_cost_volume)  # 1/8

        upsample = keras.layers.UpSampling3D(size=(2, 2, 2))
        low_to_high = upsample(low_agg_cost_volume)
        high_cost_volume += low_to_high

        high_aggregation = FactorizedCostAggregation(filters=16)
        high_agg_cost_volume = high_aggregation(high_cost_volume)

        high_computation = DisparityComputation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        high_disparity = high_computation(high_agg_cost_volume)  # 1/4

        refine = Refinement(filters=16)
        refined_disparity = refine([high_disparity, left_image])

        d0 = tf.image.resize(low_disparity, [self.height, self.width]) * 8
        d1 = tf.image.resize(high_disparity, [self.height, self.width]) * 4
        d2 = tf.image.resize(refined_disparity, [self.height, self.width]) * 2

        self.model = keras.Model(inputs=[left_image, right_image], outputs=d2)
        self.model.summary()

    def train(self, train_dir, val_dir, log_dir, weights_path, epochs, batch_size, weights):
        # all paths
        all_train_left_paths = glob.glob(train_dir + '/left/*')
        all_train_right_paths = glob.glob(train_dir + '/right/*')
        all_train_disp_paths = glob.glob(train_dir + '/disp/*')
        all_val_left_paths = glob.glob(val_dir + '/left/*')
        all_val_right_paths = glob.glob(val_dir + '/right/*')
        all_val_disp_paths = glob.glob(val_dir + '/disp/*')

        # sort, necessary in Linux
        all_train_left_paths.sort()
        all_train_right_paths.sort()
        all_train_disp_paths.sort()
        all_val_left_paths.sort()
        all_val_right_paths.sort()
        all_val_disp_paths.sort()

        # callbacks
        tb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        lr = keras.callbacks.LearningRateScheduler(schedule=schedule, verbose=1)
        mc = keras.callbacks.ModelCheckpoint(filepath=weights_path, monitor='val_tf.math.multiply_2_loss',
                                             verbose=1, save_best_only=True, save_weights_only=True,
                                             mode='min', save_freq='epoch')

        optimizer = keras.optimizers.Adam()
        loss = [SmoothL1Loss(1.0 * self.min_disp, 1.0 * self.max_disp),
                SmoothL1Loss(1.0 * self.min_disp, 1.0 * self.max_disp),
                SmoothL1Loss(1.0 * self.min_disp, 1.0 * self.max_disp)]
        loss_weights = [0.8, 1.0, 0.6]

        if weights is not None:
            self.model.load_weights(filepath=weights, by_name=True)
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)
        self.model.fit_generator(
            generator=load_batch(all_train_left_paths, all_train_right_paths, all_train_disp_paths, batch_size, True),
            steps_per_epoch=len(all_train_disp_paths) // batch_size, epochs=epochs, callbacks=[tb, lr, mc],
            validation_data=load_batch(all_val_left_paths, all_val_right_paths, all_val_disp_paths, 1, False),
            validation_steps=len(all_val_disp_paths), shuffle=False)

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
        mc = keras.callbacks.ModelCheckpoint(filepath=weights_save_path, monitor='tf.math.multiply_2_loss',
                                             verbose=1, save_best_only=True, save_weights_only=True,
                                             mode='min', save_freq='epoch')

        optimizer = keras.optimizers.Adam()
        loss = [SmoothL1Loss(1.0 * self.min_disp, 1.0 * self.max_disp),
                SmoothL1Loss(1.0 * self.min_disp, 1.0 * self.max_disp),
                SmoothL1Loss(1.0 * self.min_disp, 1.0 * self.max_disp)]
        loss_weights = [0.8, 1.0, 0.6]

        if pre_trained_weights is not None:
            self.model.load_weights(filepath=pre_trained_weights, by_name=True)
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)
        self.model.fit_generator(
            generator=load_batch(all_left_paths, all_right_paths, all_disp_paths, batch_size, True),
            steps_per_epoch=len(all_disp_paths) // batch_size, epochs=epochs, callbacks=[lr, mc],
            shuffle=False)

    def predict(self, left_dir, right_dir, output_dir, weights):
        self.model.load_weights(weights)
        lefts = os.listdir(left_dir)
        rights = os.listdir(right_dir)
        lefts.sort()
        rights.sort()
        assert len(lefts) == len(rights)
        for left, right in zip(lefts, rights):
            left_image = np.expand_dims(read_image(os.path.join(left_dir, left)), 0)
            right_image = np.expand_dims(read_image(os.path.join(right_dir, right)), 0)
            disparity = self.model.predict([left_image, right_image])
            disparity = Image.fromarray(disparity[-1][0, :, :, 0])
            name = left.replace('left', 'disparity')
            disparity.save(os.path.join(output_dir, name))
