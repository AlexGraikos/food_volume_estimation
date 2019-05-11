import tensorflow as tf
import keras.backend as K
from keras.layers import Layer

from project import *

class ProjectionLayer(Layer):
    """
    Wraps the projective inverse warp utility function as a Keras layer
    Initialize with a camera intrinsics matrix which is kept constant
    during training
    """

    def __init__(self, intrinsicsMatrix=None, **kwargs):
        if intrinsicsMatrix is None:
            self.intrinsicsMatrix = np.array([[1, 0, 0.5],
                [0, 1, 0.5], [0, 0, 1]])
        else:
            self.intrinsicsMatrix = intrinsicsMatrix
        self.intrinsicsMatrixInverse = np.linalg.inv(self.intrinsicsMatrix)
        super(ProjectionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.intrinsicsTensor = K.variable(self.intrinsicsMatrix)
        self.intrinsicsInverseTensor = K.variable(self.intrinsicsMatrixInverse)
        super(ProjectionLayer, self).build(input_shape)

    def call(self, x):
        source_img = x[0]
        depth_map = x[1]
        pose = x[2]*0.01
        reprojected_img, _ = inverse_warp(source_img, depth_map, pose,
            self.intrinsicsTensor, self.intrinsicsInverseTensor)
        return reprojected_img

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class ReflectionPadding2D(Layer):
    """
    Reflection padding layer
    """
    def __init__(self, padding=(1,1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, x):
        return tf.pad(x, [[0,0], [self.padding[0], self.padding[0]],
            [self.padding[1], self.padding[1]], [0,0]], 'REFLECT')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]+2*self.padding[0],
            input_shape[2]+2*self.padding[1], input_shape[3])


def inverseDepthNormalization(disparityMap):
    """
    Normalizes and inverses given disparity map
        Inputs:
            disparityMap: Network-generated disparity map to be normalized
        Outputs:
            depthMap: The corresponding depth map
    """
    max_depth = 10
    min_depth = 0.1
    normalizedDisp = 1/max_depth + (1/min_depth - 1/max_depth)*disparityMap
    depthMap = 1 / normalizedDisp
    return depthMap


def perScaleMinMAE(y_true, y_pred):
    """
    Computes the minimum mean absolute error between the target image and 
    the source to target reprojections
        Inputs:
            y_true: Target image
            y_pred: Source to target reprojections concatenated along the channel axis
        Outputs:
            min_error: Minimum MAE between the two reprojections
    """
    # Split channels to separate inputs
    #prev_frame = y_pred[:,:,:,:3]
    #next_frame = y_pred[:,:,:,3:6]
    reprojection_prev = y_pred[:,:,:,:3]
    reprojection_next = y_pred[:,:,:,3:]
    # MAE
    #mae_prev = K.mean(K.abs(y_true-prev_frame), axis=-1, keepdims=True)
    #mae_next = K.mean(K.abs(y_true-next_frame), axis=-1, keepdims=True)
    #minMAE = K.minimum(mae_prev, mae_next)
    mae_reprojection_prev = K.mean(K.abs(y_true-reprojection_prev), axis=-1, keepdims=True)
    mae_reprojection_next = K.mean(K.abs(y_true-reprojection_next), axis=-1, keepdims=True)
    minMAE_reprojection = K.minimum(mae_reprojection_prev, mae_reprojection_next)
    # SSIM
    ssim_reprojection_prev = ssim(reprojection_prev, y_true)
    ssim_reprojection_next = ssim(reprojection_next, y_true)
    minSSIM_reprojection = K.minimum(ssim_reprojection_prev, ssim_reprojection_next)

    #mask = K.less(minMAE_reprojection, minMAE)
    #minMAE_reprojection *= K.cast(mask, 'float32')
    return minMAE_reprojection


def perScaleMinSSIM(y_true, y_pred):
    # Split channels to separate inputs
    #prev_frame = y_pred[:,:,:,:3]
    #next_frame = y_pred[:,:,:,3:6]
    reprojection_prev = y_pred[:,:,:,:3]
    reprojection_next = y_pred[:,:,:,3:]
    # SSIM
    ssim_reprojection_prev = ssim(reprojection_prev, y_true)
    ssim_reprojection_next = ssim(reprojection_next, y_true)
    minSSIM_reprojection = K.minimum(ssim_reprojection_prev, ssim_reprojection_next)

    return minSSIM_reprojection


def ssim(x, y):
    """
    Computes a differentiable structured image similarity measure
    Taken from https://github.com/tensorflow/models/tree/master/research/struct2depth
    """
    c1 = 0.01**2  # As defined in SSIM to stabilize div. by small denominator.
    c2 = 0.03**2
    mu_x = tf.contrib.slim.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = tf.contrib.slim.avg_pool2d(y, 3, 1, 'VALID')
    sigma_x = tf.contrib.slim.avg_pool2d(x**2, 3, 1, 'VALID') - mu_x**2
    sigma_y = tf.contrib.slim.avg_pool2d(y**2, 3, 1, 'VALID') - mu_y**2
    sigma_xy = tf.contrib.slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    return tf.clip_by_value((1 - ssim) / 2, 0, 1)
