import tensorflow as tf
import keras.backend as K
from keras.layers import Layer
from project import *

class ProjectionLayer(Layer):
    """
    Projective inverse warping layer.
    Initialize with the camera intrinsics matrix which is kept constant
    during training.
    """
    def __init__(self, intrinsics_mat=None, **kwargs):
        self.POSE_SCALING = 0.01
        if intrinsics_mat is None:
            self.intrinsics_mat = np.array([[1, 0, 0.5],
                                            [0, 1, 0.5],
                                            [0, 0,   1]])
        else:
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


class ReflectionPadding2D(Layer):
    """
    Reflection padding layer.
    Padding (p1,p2) is applied as ([p1 rows p1], [p2 cols p2]).
    """
    def __init__(self, padding=(1,1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, x):
        return tf.pad(x, [[0,0], [self.padding[0], self.padding[0]],
                          [self.padding[1], self.padding[1]], [0,0]],
                      'REFLECT')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]+2*self.padding[0],
                input_shape[2]+2*self.padding[1], input_shape[3])


def normalize_inverse_depth(disparityMap):
    """
    Normalizes and inverses disparities to create depth map with
    given max and min depth values.
        Inputs:
            disparityMap: Network-generated disparity map to be normalized.
        Outputs:
            depthMap: The corresponding depth map.
    """
    MAX_DEPTH = 10
    MIN_DEPTH = 0.1
    MAX_DISP = 1 / MIN_DEPTH
    MIN_DISP = 1 / MAX_DEPTH

    normalizedDisp = MIN_DISP + (MAX_DISP - MIN_DISP) * disparityMap
    depthMap = 1 / normalizedDisp
    return depthMap


def per_scale_MAE(y_true, y_pred):
    """
    Computes the minimum MAE between the target image and the source
    to target reprojections of scale s.
        Inputs:
            y_true: Target image.
            y_pred: Source to target reprojections concatenated along 
                    the channel axis.
        Outputs:
            scale_min_mae: Minimum MAE between the two reprojections.
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
    reprojection_prev_mae = K.mean(K.abs(y_true - reprojection_prev),
                                   axis=-1, keepdims=True)
    reprojection_next_mae = K.mean(K.abs(y_true - reprojection_next),
                                   axis=-1, keepdims=True)
    scale_min_mae = K.minimum(reprojection_prev_mae, reprojection_next_mae)

    #mask = K.less(minMAE_reprojection, minMAE)
    #minMAE_reprojection *= K.cast(mask, 'float32')
    return scale_min_mae 


def per_scale_SSIM(y_true, y_pred):
    """
    Computes the minimum SSIM between the target image and the source
    to target reprojections of scale s.
        Inputs:
            y_true: Target image.
            y_pred: Source to target reprojections concatenated along 
                    the channel axis.
        Outputs:
            scale_min_ssim: Minimum SSIM between the two reprojections.
    """
    # Split channels to separate inputs
    #prev_frame = y_pred[:,:,:,:3]
    #next_frame = y_pred[:,:,:,3:6]
    reprojection_prev = y_pred[:,:,:,:3]
    reprojection_next = y_pred[:,:,:,3:]
    # SSIM
    reprojection_prev_ssim = ssim(reprojection_prev, y_true)
    reprojection_next_ssim = ssim(reprojection_next, y_true)
    scale_min_ssim = K.minimum(reprojection_prev_ssim, reprojection_next_ssim)

    return scale_min_ssim


def ssim(x, y):
    """
    Computes a differentiable structured image similarity measure
    Taken from:
        https://github.com/tensorflow/models/tree/master/research/struct2depth
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
