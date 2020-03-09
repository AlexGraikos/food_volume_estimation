import json
import pandas as pd
import cv2
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer
from keras.utils import Sequence
from food_volume_estimation.depth_estimation.project import *


class AugmentationLayer(Layer):
    """Image batch augmentation layer. Random color augmentations are 
    applied to input images during training, with given probability.
    Augmentation seeds are created to perform the same transformations
    on all three input frames for consistency.
    """
    def __init__(self, augment_prob=0.5, brightness_range=0.2,
            contrast_range=0.2, saturation_range=0.2, hue_range=0.1,
            **kwargs):
        self.augment_prob = augment_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        super(AugmentationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.augment_prob_tensor = tf.constant(self.augment_prob)
        super(AugmentationLayer, self).build(input_shape)

    def call(self, x):
        def transform():
            # Apply transformations with given probability
            p = tf.random.uniform([], 0.0, 1.0)
            x_transformed = tf.cond(tf.math.less(p, self.augment_prob_tensor),
                                    lambda: self.__augment_inputs(x),
                                    lambda: x)
            return x_transformed

        # Do not apply augmentations during testing
        y = tf.cond(K.learning_phase(), lambda: transform(), lambda: x)
        return y
        
    def __augment_inputs(self, x):
        """Input augmentation function.

        Inputs:
            x: Input frame triplet. 
        Outputs:
            aug_x: Augmented input frames.
        """
        # Unpack inputs
        curr_frame = x[0]
        prev_frame = x[1]
        next_frame = x[2]

        # Brightness
        brightness_seed = int(np.random.rand() * 1e6)
        curr_frame = tf.image.random_brightness(
            curr_frame, self.brightness_range, seed=brightness_seed)
        prev_frame = tf.image.random_brightness(
            prev_frame, self.brightness_range, seed=brightness_seed)
        next_frame = tf.image.random_brightness(
            next_frame, self.brightness_range, seed=brightness_seed)
        # Contrast
        contrast_seed = int(np.random.rand() * 1e6)
        curr_frame = tf.image.random_contrast(
            curr_frame, 1 - self.contrast_range, 1 + self.contrast_range,
            seed=contrast_seed)
        prev_frame = tf.image.random_contrast(
            prev_frame, 1 - self.contrast_range, 1 + self.contrast_range,
            seed=contrast_seed)
        next_frame = tf.image.random_contrast(
            next_frame, 1 - self.contrast_range, 1 + self.contrast_range,
            seed=contrast_seed)
        # Saturation
        saturation_seed = int(np.random.rand() * 1e6)
        curr_frame = tf.image.random_saturation(
            curr_frame, 1 - self.saturation_range,
            1 + self.saturation_range, seed=saturation_seed)
        prev_frame = tf.image.random_saturation(
            prev_frame, 1 - self.saturation_range,
            1 + self.saturation_range, seed=saturation_seed)
        next_frame = tf.image.random_saturation(
            next_frame, 1 - self.saturation_range,
            1 + self.saturation_range, seed=saturation_seed)
        # Hue
        hue_seed = int(np.random.rand() * 1e6)
        curr_frame = tf.image.random_hue(curr_frame, self.hue_range,
                                         seed=hue_seed)
        prev_frame = tf.image.random_hue(prev_frame, self.hue_range,
                                         seed=hue_seed)
        next_frame = tf.image.random_hue(next_frame, self.hue_range,
                                         seed=hue_seed)

        return [curr_frame, prev_frame, next_frame]

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'augment_prob': self.augment_prob,
            'brightness_range': self.brightness_range,
            'contrast_range':  self.contrast_range,
            'saturation_range': self.saturation_range,
            'hue_range': self.hue_range
        }
        base_config = super(AugmentationLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ProjectionLayer(Layer):
    """Projective inverse warping layer. Initialize with the camera 
    intrinsics matrix which is kept constant during training.
    """
    def __init__(self, intrinsics_mat=None, **kwargs):
        self.POSE_SCALING = 0.001
        self.intrinsics_mat = intrinsics_mat
        self.intrinsics_mat_inv = np.linalg.inv(self.intrinsics_mat)
        super(ProjectionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.intrinsics_mat_tensor = K.variable(self.intrinsics_mat)
        self.intrinsics_mat_inv_tensor = K.variable(self.intrinsics_mat_inv)
        super(ProjectionLayer, self).build(input_shape)

    def call(self, x):
        source_img = x[0]
        depth_map = x[1]
        pose = x[2] * self.POSE_SCALING
        reprojected_img, _ = inverse_warp(source_img, depth_map, pose,
                                          self.intrinsics_mat_tensor,
                                          self.intrinsics_mat_inv_tensor)
        return reprojected_img

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {
            'intrinsics_mat': self.intrinsics_mat
        }
        base_config = super(ProjectionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ReflectionPadding2D(Layer):
    """Reflection padding layer. Padding (p1,p2) is applied as 
    ([p1 rows p1], [p2 cols p2]).
    """
    def __init__(self, padding=(1,1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, x):
        return tf.pad(x, [[0,0], [self.padding[0], self.padding[0]],
                          [self.padding[1], self.padding[1]], [0,0]],
                      'REFLECT')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + (2 * self.padding[0]),
                input_shape[2] + (2 * self.padding[1]), input_shape[3])

    def get_config(self):
        config = {
            'padding': self.padding
        }
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class InverseDepthNormalization(Layer):
    """Normalizes and inverses disparities to create depth map with
    given max and min values.
    """
    def __init__(self, min_depth=0.01, max_depth=10, **kwargs):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_disp = 1 / max_depth
        self.max_disp = 1 / min_depth
        super(InverseDepthNormalization, self).__init__(**kwargs)

    def call(self, x):
        normalized_disp = (self.min_disp
                           + (self.max_disp - self.min_disp) * x)
        depth_map = 1 / normalized_disp
        return depth_map

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'min_depth': self.min_depth,
            'max_depth': self.max_depth
        }
        base_config = super(InverseDepthNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Losses():
    def reprojection_loss(self, alpha=0.85, masking=True):
        """Creates reprojection loss function combining MAE and SSIM losses.
        The reprojection loss is computed per scale by choosing the minimum
        loss between the previous and next frame reprojections.

        Inputs:
            alpha: SSIM loss weight
        Outputs:
            reprojection_loss: Reprojection Keras-style loss function
        """
        def reprojection_loss_keras(y_true, y_pred):
            source_loss = y_pred[:,:,:,:3]
            reprojection_prev = y_pred[:,:,:,3:6]
            reprojection_next = y_pred[:,:,:,6:9]

            # Reprojection MAE
            reprojection_prev_mae = K.mean(K.abs(y_true - reprojection_prev),
                                           axis=-1, keepdims=True)
            reprojection_next_mae = K.mean(K.abs(y_true - reprojection_next),
                                           axis=-1, keepdims=True)
            scale_min_mae = K.minimum(reprojection_prev_mae, 
                                      reprojection_next_mae)
            # Reprojection SSIM
            reprojection_prev_ssim = self.__ssim(y_true, reprojection_prev)
            reprojection_next_ssim = self.__ssim(y_true, reprojection_next)
            scale_min_ssim = K.minimum(reprojection_prev_ssim,
                                       reprojection_next_ssim)
            # Total loss
            reprojection_loss = (alpha * scale_min_ssim 
                                 + (1 - alpha) * scale_min_mae)
            if masking:
                mask = K.less(reprojection_loss, source_loss)
                reprojection_loss *= K.cast(mask, 'float32')

            return reprojection_loss

        return reprojection_loss_keras

    def compute_source_loss(self, x, alpha=0.85):
        """Compute minimum reprojection loss using the prev and next frames
        as reprojections.
        """
        y_true = x[0]
        prev_frame = x[1]
        next_frame = x[2]

        # Source frame MAE
        prev_mae = K.mean(K.abs(y_true - prev_frame), axis=-1,
                          keepdims=True)
        next_mae = K.mean(K.abs(y_true - next_frame), axis=-1,
                          keepdims=True)
        source_min_mae  = K.minimum(prev_mae, next_mae)
        # Source frame SSIM
        prev_ssim = self.__ssim(y_true, prev_frame)
        next_ssim = self.__ssim(y_true, next_frame)
        source_min_ssim = K.minimum(prev_ssim, next_ssim)
        source_loss = (alpha * source_min_ssim
                       + (1 - alpha) * source_min_mae)
        return source_loss

    def depth_smoothness(self):
        """Computes image-aware depth smoothness loss.
        Taken from:
            https://github.com/tensorflow/models/tree/master/research/struct2depth
        Modified by Alexander Graikos.
        """
        def depth_smoothness_keras(y_true, y_pred):
            img = y_true
            # Normalize inverse depth by mean
            inverse_depth = y_pred / (tf.reduce_mean(y_pred, axis=[1,2,3], 
                                      keepdims=True) + 1e-7)
            # Compute depth smoothness loss
            inverse_depth_dx = self.__gradient_x(inverse_depth)
            inverse_depth_dy = self.__gradient_y(inverse_depth)
            image_dx = self.__gradient_x(img)
            image_dy = self.__gradient_y(img)
            weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_dx), 3, 
                                               keepdims=True))
            weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_dy), 3,
                                               keepdims=True))
            smoothness_x = inverse_depth_dx * weights_x
            smoothness_y = inverse_depth_dy * weights_y
            return (tf.reduce_mean(tf.abs(smoothness_x)) 
                    + tf.reduce_mean(tf.abs(smoothness_y)))

        return depth_smoothness_keras

    def __ssim(self, x, y):
        """Computes a differentiable structured image similarity measure.
        Taken from:
            https://github.com/tensorflow/models/tree/master/research/struct2depth
        Modified by Alexander Graikos.
        """
        c1 = 0.01**2  # As defined in SSIM to stabilize div. by small denom.
        c2 = 0.03**2
        # Add padding to maintain img size
        x = tf.pad(x, [[0,0], [1,1], [1,1], [0,0]], 'REFLECT')
        y = tf.pad(y, [[0,0], [1,1], [1,1], [0,0]], 'REFLECT')
        mu_x = K.pool2d(x, (3,3), (1,1), 'valid', pool_mode='avg')
        mu_y = K.pool2d(y, (3,3), (1,1), 'valid', pool_mode='avg')
        sigma_x = (K.pool2d(x**2, (3,3), (1,1), 'valid', pool_mode='avg')
                   - mu_x**2)
        sigma_y = (K.pool2d(y**2, (3,3), (1,1), 'valid', pool_mode='avg')
                   - mu_y**2)
        sigma_xy = (K.pool2d(x * y, (3,3), (1,1), 'valid', pool_mode='avg') 
                    - mu_x * mu_y)
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
        ssim = ssim_n / ssim_d
        return K.clip((1 - ssim) / 2, 0, 1)

    def __gradient_x(self, img):
        return img[:, :, :-1, :] - img[:, :, 1:, :]

    def __gradient_y(self, img):
        return img[:, :-1, :, :] - img[:, 1:, :, :]


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
                              np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class DataGenerator(Sequence):
    """Generates batches of frame triplets, with given size."""
    def __init__(self, data_df, height, width, batch_size, flipping):
        self.data_df = data_df
        self.target_size = (width, height)
        self.batch_size = batch_size
        self.flipping = flipping
        print('[*] Found', self.data_df.shape[0], 'frame triplets.')
        np.random.shuffle(self.data_df.values)

    def __len__(self):
        num_samples = self.data_df.shape[0]
        return num_samples // self.batch_size

    def on_epoch_end(self):
        # Shuffle data samples on epoch end
        np.random.shuffle(self.data_df.values)

    def __read_img(self, path, flip):
        """Load input image and flip if specified.

        Inputs:
            path: Path to input image.
            flip: Flipping flag.
        Outputs:
            img: Loaded image with pixel values [0,1].
        """
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.flipping and flip:
            img = cv2.flip(img, 1)
        img = cv2.resize(img, self.target_size,
                         interpolation=cv2.INTER_LINEAR)
        return img.astype(np.float32) / 255

    def __getitem__(self, idx):
        """Generate and return network training data.
        
        Inputs:
            idx: Index of current batch.
        Outputs:
            ([inputs], [outputs]) tuple for model training.
        """
        # Load file paths to current batch images
        curr_batch_fp = self.data_df.iloc[
            idx * self.batch_size : (idx + 1) * self.batch_size, 0].values
        prev_batch_fp = self.data_df.iloc[
            idx * self.batch_size : (idx + 1) * self.batch_size, 1].values
        next_batch_fp = self.data_df.iloc[
            idx * self.batch_size : (idx + 1) * self.batch_size, 2].values

        # Load and flip horizontally with probability 0.5
        horizontal_flips = np.random.rand(self.batch_size) > 0.5
        curr_frame = np.array(
            [self.__read_img(curr_batch_fp[i], horizontal_flips[i])
             for i in range(curr_batch_fp.shape[0])])
        prev_frame = np.array(
            [self.__read_img(prev_batch_fp[i], horizontal_flips[i])
             for i in range(prev_batch_fp.shape[0])])
        next_frame = np.array(
            [self.__read_img(next_batch_fp[i], horizontal_flips[i])
             for i in range(next_batch_fp.shape[0])])

        # Return (inputs,outputs) tuple
        curr_frame_list = [curr_frame for _ in range(4)]
        curr_frame_scales = []
        for s in [1, 2, 4, 8]:
            curr_frame_scales += [curr_frame[:,::s,::s,:]]

        return ([curr_frame, prev_frame, next_frame],
                (curr_frame_list + curr_frame_scales))

