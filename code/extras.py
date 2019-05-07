import keras.backend as K
from keras.layers import Layer
from utils import *

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
        super(ProjectionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.intrinsicsTensor = K.variable(self.intrinsicsMatrix)
        super(ProjectionLayer, self).build(input_shape)

    def call(self, x):
        source_img = x[0]
        depth_map = x[1]
        # Scale translation and rotation
        translation = x[2][:,:3]*0.01
        rotation = x[2][:,3:]*0.001
        pose_vector = K.concatenate([translation, rotation], axis=-1)
        return projective_inverse_warp(source_img, depth_map, pose_vector, self.intrinsicsTensor)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def inverseDepthNormalization(disparityMap):
    """
    Normalizes and inverses given disparity map
        Inputs:
            disparityMap: Network-generated disparity map to be normalized
        Outputs:
            depthMap: The corresponding depth map
    """
    epsilon = 10e-6
    disparityMap = 10*disparityMap + epsilon # To avoid division by zero

    mean = K.mean(disparityMap, axis=[1,2,3], keepdims=True)
    normalizedDisp = disparityMap / mean
    depthMap = 1 / normalizedDisp
    return depthMap


def generateAdversarialInput(input_images, omega):
    """
    Generates an adversarial input image based on the target and reprojected image differences
        Inputs:
            input_images: Target and reprojected image tensors
        Outputs:
            adversarial_input: The created adversarial image
    """
    target_img = input_images[0]
    reprojected_img = input_images[1]

    difference = K.abs(target_img - reprojected_img)
    noise_img = reprojected_img*difference
    adversarial_input = omega*reprojected_img + (1-omega)*noise_img
    return adversarial_input


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
    # Split channels to prev and next reprojection
    reprojection_prev = y_pred[:,:,:,:3]
    reprojection_next = y_pred[:,:,:,3:]

    mae_prev = K.sum(K.abs(y_true-reprojection_prev), axis=-1, keepdims=True)
    mae_next = K.sum(K.abs(y_true-reprojection_next), axis=-1, keepdims=True)
    minMAE = K.minimum(mae_prev, mae_next)
    return K.mean(minMAE, axis=[1,2,3])


"""
Taken from
https://github.com/tensorflow/models/tree/master/research/struct2depth
Modified
"""
def gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def gradient_y(img):
    return img[:, :-1, :, :] - img[:, 1:, :, :]


def depth_smoothness(y_true, y_pred):
    """
    Computes image-aware depth smoothness loss
        Inputs:
            y_true: Target image
            y_pred: Predicted depth map
        Outputs:
            smoothness_loss: Depth smoothness loss
    """
    depth = y_pred
    img= y_true
    img_resized = tf.image.resize_nearest_neighbor(img, depth.shape[1:3])

    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)
    image_dx = gradient_x(img_resized)
    image_dy = gradient_y(img_resized)
    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_dx), 3, keepdims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_dy), 3, keepdims=True))
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))
    return smoothness_loss

