import argparse
import numpy as np
import pandas as pd
import cv2
import json
from keras.models import Model, model_from_json
import keras.backend as K
from custom_modules import *
from project import *
from point_cloud_utils import *
import matplotlib.pyplot as plt


class VolumeEstimator():
    """
    Volume estimator class.
    """

    def __init__(self, arg_init=True):
        """
        Load depth model and segmentation module.
        """
        if not arg_init:
            # For usage in jupyter notebook 
            print('[*] VolumeEstimator not initialized.')
        else:    
            self.args = self.__parse_args()
            # Load depth estimation model
            custom_losses = Losses()
            objs = {'ProjectionLayer': ProjectionLayer,
                    'ReflectionPadding2D': ReflectionPadding2D,
                    'InverseDepthNormalization': InverseDepthNormalization,
                    'AugmentationLayer': AugmentationLayer,
                    'compute_source_loss': custom_losses.compute_source_loss}
            with open(self.args.depth_model_architecture, 'r') as read_file:
                model_architecture_json = json.load(read_file)
                self.monovideo = model_from_json(model_architecture_json,
                                                 custom_objects=objs)
            self.__set_weights_trainable(self.monovideo, False)
            self.monovideo.load_weights(self.args.depth_model_weights)
            self.model_input_shape = self.monovideo.inputs[0].shape[1:]
            depth_net = self.monovideo.get_layer('depth_net')
            self.depth_model = Model(inputs=depth_net.inputs,
                                     outputs=depth_net.outputs,
                                     name='depth_model')
            print('[*] Loaded depth estimation model.')
            # Depth model configuration
            self.GT_DEPTH_SCALE = self.args.gt_depth_scale
            self.min_disp = 1 / self.args.max_depth
            self.max_disp = 1 / self.args.min_depth

            # Select segmentation model to use
            self.__select_segmentation_module(self.args.segmentation_model)
            print('[*] Selected', self.args.segmentation_model,
                  'as the segmentation model.')


    def __parse_args(self):
        """
        Parse command-line input arguments.
            Outputs:
                args: The arguments object.
        """
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description='Estimate food volume in input image.')
        parser.add_argument('--input_image', type=str,
                            help='Input image path.',
                            default=None)
        parser.add_argument('--depth_model_architecture', type=str,
                            help=('Depth estimation model '
                                  + 'architecture (.json).'),
                            default=None)
        parser.add_argument('--depth_model_weights', type=str,
                            help='Depth estimation model weights (.h5).',
                            default=None)
        parser.add_argument('--segmentation_model', type=str,
                            help='Food segmentation model [GAP/GMAP/GMP].',
                            default='GAP')
        parser.add_argument('--fov', type=float,
                            help='Camera Field of View (in deg).',
                            default=None)
        parser.add_argument('--focal_length', type=float,
                            help='Camera focal length (in px).',
                            default=None)
        parser.add_argument('--gt_depth_scale', type=float,
                            help='Ground truth depth rescaling factor.',
                            default=1)
        parser.add_argument('--min_depth', type=float,
                            help='Minimum depth value.',
                            default=0.01)
        parser.add_argument('--max_depth', type=float,
                            help='Maximum depth value.',
                            default=10)
        parser.add_argument('--plot_results', action='store_true',
                            help='Plot volume estimation results.',
                            default=False)
        args = parser.parse_args()
        return args


    def estimate_volume(self, input_image, fov=None, focal_length=None, 
            plot_results=False):
        """
        Volume estimation pipeline.
            Inputs:
                input_image: Path to input image.
                fov: Camera Field of View.
                focal_length: Camera Focal length.
                plot_results: Result plotting flag.
            Outputs:
                estimated_volume: Estimated volume.
        """
        # Load input image and resize to model input size
        img = cv2.imread(input_image, cv2.IMREAD_COLOR)
        input_image_shape = img.shape
        img = cv2.resize(img, (self.model_input_shape[1],
                               self.model_input_shape[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255

        # Predict segmentation mask
        object_mask = self.food_segmentation(
            input_image, self.segmentation_weights)
        object_mask = ((cv2.resize(
            object_mask, (self.model_input_shape[1],
            self.model_input_shape[0])) / 255) >= 0.5)
        # Predict depth
        img_batch = np.reshape(img, (1,) + img.shape)
        inverse_depth = self.depth_model.predict(img_batch)[0][0,:,:,0] 
        disparity_map = (self.min_disp + (self.max_disp - self.min_disp) 
                           * inverse_depth)
        predicted_median_depth = np.median(1 / disparity_map)
        depth = ((self.GT_DEPTH_SCALE / predicted_median_depth) 
                 * (1 / disparity_map))
        # Apply mask to create object image and depth map
        object_colors = (np.tile(np.expand_dims(object_mask, axis=-1),
                                 (1,1,3)) * img)
        object_depth = object_mask * depth

        # Create intrinsics matrix
        intrinsics_mat = self.__create_intrinsics_matrix(
            input_image_shape, fov, focal_length)
        intrinsics_inv = np.linalg.inv(intrinsics_mat)
        # Convert depth map to point cloud
        depth_tensor = K.variable(np.expand_dims(depth, 0))
        intrinsics_inv_tensor = K.variable(np.expand_dims(intrinsics_inv, 0))
        point_cloud = K.eval(get_cloud(depth_tensor, intrinsics_inv_tensor))
        point_cloud_flat = np.reshape(
            point_cloud, (point_cloud.shape[1] * point_cloud.shape[2], 3))
        # Get object points by filtering zero depth pixels
        object_filter = (np.reshape(
            object_depth, (object_depth.shape[0] * object_depth.shape[1]))
            > 0)
        object_points = point_cloud_flat[object_filter, :]

        # Estimate plate plane parameters and filter outlier object points
        plane_params = ransac_plane_estimation(
            object_points, k=(0.05*object_points.shape[0]))
        object_points_filtered = sor_filter(object_points, 2, 0.7)
        # Transform object to match z-axis with plate normal
        translation, rotation_matrix = align_plane_with_axis(
            plane_params, np.array([0, 0, 1]))
        object_points_transformed = np.dot(
            object_points_filtered + translation, rotation_matrix.T)
        # Move all object points above the estimated plane
        object_points_transformed[:,2] += np.abs(np.min(
            object_points_transformed[:,2]))

        if plot_results:
            # Create all-points and object points dataFrames
            colors_flat = (np.reshape(
                img, (self.model_input_shape[0] * self.model_input_shape[1],
                3)) * 255)
            object_colors_flat = colors_flat[object_filter, :]
            all_points_df = pd.DataFrame(
                np.concatenate((point_cloud_flat, colors_flat), axis=-1),
                columns=['x','y','z','red','green','blue'])
            object_points_df = pd.DataFrame(
                np.concatenate((object_points, object_colors_flat), axis=-1),
                columns=['x','y','z','red','green','blue'])
            # Create estimated plane points dataFrame
            plane_z = np.apply_along_axis(
                lambda x: ((plane_params[0] + plane_params[1] * x[0]
                + plane_params[2] * x[1]) * (-1) / plane_params[3]),
                axis=1, arr=all_points_df.values[:,:2])
            plane_points = np.concatenate(
                (all_points_df.values[:,:2], np.expand_dims(plane_z, axis=-1)), axis=-1)
            plane_points_df = pd.DataFrame(plane_points,
                                           columns=['x','y','z'])
            # Create transformed object and plane points dataFrames
            object_points_transformed_df = pd.DataFrame(
                object_points_transformed, columns=['x','y','z'])
            plane_points_transformed = np.dot(plane_points + translation, 
                                              rotation_matrix.T)
            plane_points_transformed_df = pd.DataFrame(
                plane_points_transformed, columns=['x','y','z'])
            print('[*] Estimated plane parameters (w0,w1,w2,w3):',
                  plane_params)

            # Estimate volume
            estimated_volume, simplices = pc_to_volume(
                object_points_transformed)
            print('[*] Estimated volume:', estimated_volume * 1000, 'L')

            # Plot input image and predicted segmentation mask/depth
            pretty_plotting([img, depth, object_colors, object_depth], (2,2),
                            ['Input Image', 'Depth', 'Object Mask',
                             'Object Depth'])
            plt.show()

            return (estimated_volume, object_points_df, all_points_df,
                    plane_points_df, object_points_transformed_df,
                    plane_points_transformed_df, simplices)
        else:
            # Estimate volume
            estimated_volume, _ = pc_to_volume(object_points_transformed)
            return estimated_volume


    def __create_intrinsics_matrix(self, input_image_shape, fov,
            focal_length):
        """
        Create intrinsics matrix from given parameters or return default
        if none given.
            Inputs:
                input_image_shape: Original input image shape.
                fov: Camera Field of View (in deg).
                focal_length: Camera focal length (in px).
            Outputs:
                intrinsics_mat: Intrinsics matrix [3x3].
        """
        if fov is not None:
            F = input_image_shape[1] / (2 * np.tan((fov / 2) * np.pi / 180))
            print('[*] Creating intrinsics matrix from given FOV:', fov)
        elif focal_length is not None:
            F = focal_length
            print('[*] Creating intrinsics matrix from given focal length:',
                  focal_length)
        else:
            fov = 60 # Default value
            F = input_image_shape[1] / (2 * np.tan((fov / 2) * np.pi / 180))

        # Create intrinsics matrix
        x_scaling = int(self.model_input_shape[1]) / input_image_shape[1] 
        y_scaling = int(self.model_input_shape[0]) / input_image_shape[0] 
        intrinsics_mat = np.array(
            [[F * x_scaling, 0, (input_image_shape[1] / 2) * x_scaling], 
             [0, F * y_scaling, (input_image_shape[0] / 2) * y_scaling],
             [0, 0, 1]])
        return intrinsics_mat


    def __select_segmentation_module(self, segmentation_model):
        """
        Select the food segmentation module from the available source files.
            Inputs:
                segmentation_model: Segmentation model to use [GAP/GMAP/GMP].
        """
        if segmentation_model == 'GAP':
            from segmentation.gap_segmentation import get_food_segmentation
            self.segmentation_weights = (
                'segmentation/weights.epoch-15-val_loss-0.85.hdf5')
        elif segmentation_model == 'GMAP':
            from segmentation.gmap_segmentation import get_food_segmentation
            self.segmentation_weights = (
                'segmentation/weights.epoch-21-val_loss-0.85.hdf5')
        elif segmentation_model == 'GMP':
            from segmentation.gmp_segmentation import get_food_segmentation
            self.segmentation_weights = (
                'segmentation/weights.epoch-19-val_loss-0.79.hdf5')
        else:
            print('[!] Unknown segmentation model.')
            exit()
        self.food_segmentation = get_food_segmentation


    def __set_weights_trainable(self, model, trainable):
        """
        Sets model weights to trainable/non-trainable.
            Inputs:
                model: Model to set weights.
                trainable: Trainability flag.
        """
        for layer in model.layers:
            layer.trainable = trainable
            if isinstance(layer, Model):
                self.__set_weights_trainable(layer, trainable)



if __name__ == '__main__':
    estimator = VolumeEstimator()
    estimator.estimate_volume(estimator.args.input_image, estimator.args.fov,
                              estimator.args.focal_length,
                              estimator.args.plot_results)

