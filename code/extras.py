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
        rotation = x[2][:,3:]*0.01
        pose_vector = K.concatenate([translation, rotation], axis=-1)
        return projective_inverse_warp(source_img, depth_map, pose_vector, self.intrinsicsTensor)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def dispActivation(x):
    """
    Custom activation function for inverse depth layers
        Inptus:
            x: Inverse depth input tensor
        Outputs:
            y: a*sigmoid(x) + b
    """
    # Calculate alpha, beta from inverse max and min values
    depthMaxVal = 10
    depthMinVal = 0.01
    beta = 1/depthMaxVal
    alpha = (1/depthMinVal) - beta
    
    return alpha*K.sigmoid(x) + beta 


def inverseDepthNormalization(disparityMap):
    """
    Normalizes and inverses given disparity map
        Inputs:
            disparityMap: Network-generated disparity map to be normalized
        Outputs:
            depthMap: The corresponding depth map
    """
    mean = K.mean(disparityMap, axis=[1,2,3], keepdims=True)
    normalizedDisp = disparityMap / mean
    #depthMap = 1 / normalizedDisp
    depthMap = 1 / disparityMap
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
    # Split channels to prev and next reprojection
    reprojection_prev = y_pred[:,:,:,:3]
    reprojection_next = y_pred[:,:,:,3:]

    mae_prev = K.mean(K.abs(y_true-reprojection_prev), axis=-1, keepdims=True)
    mae_next = K.mean(K.abs(y_true-reprojection_next), axis=-1, keepdims=True)
    minMAE = K.minimum(mae_prev, mae_next)
    return minMAE

