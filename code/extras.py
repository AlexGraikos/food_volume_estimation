import keras.backend as K
from keras.layers import Layer
from utils import *

class ProjectionLayer(Layer):
    """
    Wraps the projective inverse warp utility function as a Keras layer
    Initialize with a camera intrinsics matrix which is kept constant
        during training
    """

    def __init__(self, intrinsicsMatrix=np.eye(3), **kwargs):
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
    epsilon = 10e-5
    mean = K.mean(disparityMap, axis=[1,2,3], keepdims=True)
    normalizedDisp = disparityMap / mean
    depthMap = 1 / (normalizedDisp + epsilon)
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


def perReprojectionMinimumMAE(input_images):
    """
    Computes the minimum mean absolute error between the target image and 
    the source to target reprojections
        Inputs:
            input_images: [target_img, reprojection1, reprojection2] tensors
        Outputs:
            min_error: Minimum MAE between the two reprojections
    """
    target_image = input_images[0]
    reprojection1 = input_images[1]
    reprojection2 = input_images[2]

    # Compute MAE, add channels axis to concatenate and apply min
    mae1 = K.mean(K.abs(target_image-reprojection1), axis=-1, keepdims=True)
    mae2 = K.mean(K.abs(target_image-reprojection2), axis=-1, keepdims=True)
    return K.min(K.concatenate([mae1, mae2], axis=-1), axis=-1, keepdims=True)
    
    
