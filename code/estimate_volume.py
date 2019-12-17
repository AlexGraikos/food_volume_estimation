import argparse
import numpy as np
import pandas as pd
import cv2
import json
from scipy.spatial.distance import pdist
from scipy.stats import skew
from keras.models import Model, model_from_json
import keras.backend as K
from mask_rcnn_segmentation import FoodSegmentator
from custom_modules import *
from project import *
from point_cloud_utils import *
import matplotlib.pyplot as plt

# TODO: - Decide on the plane estimation. Options are:
#         a) Use ransac (or simple linear regression) on food points and then
#            correct. This is similar to finding the convex hull.
#         b) Find a way to use ransac to estimate a plate plane and calculate
#            volume according to that. Using non-object points has issues.
#       - Clean up all code. Lots of cleanup. pc_to_volume shouldn't really
#         have the point filtering. Better if it's done outside the function.

class VolumeEstimator():
    def __init__(self, arg_init=True):
        """Load depth model and create segmentator object.
        Inputs:
            arg_init: Flag to initialize volume estimator with 
                command-line arguments.
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

            # Create segmentator object
            self.segmentator = FoodSegmentator(self.args.segmentation_weights)

    def __parse_args(self):
        """Parse command-line input arguments.
        Returns:
            args: The arguments object.
        """
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description='Estimate food volume in input image.')
        parser.add_argument('--input_image', type=str,
                            help='Input image path.',
                            metavar='/path/to/image',
                            required=True)
        parser.add_argument('--depth_model_architecture', type=str,
                            help=('Depth estimation model '
                                  'architecture (.json).'),
                            metavar='/path/to/architecture.json',
                            required=True)
        parser.add_argument('--depth_model_weights', type=str,
                            help='Depth estimation model weights (.h5).',
                            metavar='/path/to/weights.h5',
                            required=True)
        parser.add_argument('--segmentation_weights', type=str,
                            help='Food segmentation model weights (.h5).',
                            metavar='/path/to/weights.h5',
                            required=True)
        parser.add_argument('--fov', type=float,
                            help='Camera Field of View (in deg).',
                            metavar='<fov>',
                            default=None)
        parser.add_argument('--focal_length', type=float,
                            help='Camera focal length (in px).',
                            metavar='<focal_length>',
                            default=None)
        parser.add_argument('--gt_depth_scale', type=float,
                            help='Ground truth depth rescaling factor.',
                            metavar='<gt_depth_scale>',
                            default=1)
        parser.add_argument('--min_depth', type=float,
                            help='Minimum depth value.',
                            metavar='<min_depth>',
                            default=0.01)
        parser.add_argument('--max_depth', type=float,
                            help='Maximum depth value.',
                            metavar='<max_depth>',
                            default=10)
        parser.add_argument('--plot_results', action='store_true',
                            help='Plot volume estimation results.',
                            default=False)
        args = parser.parse_args()
        
        assert (args.fov is not None) or (args.focal_length is not None), (
            '[!] Either the camera FoV or Focal Length must be provided')

        return args

    def estimate_volume(self, input_image, fov, focal_length, plot_results):
        """Volume estimation procedure.
        Inputs:
            input_image: Path to input image.
            fov: Camera Field of View.
            focal_length: Camera Focal length.
            plot_results: Result plotting flag.
        Returns:
            estimated_volume: Estimated volume.
        """
        # Load input image and resize to model input size
        img = cv2.imread(input_image, cv2.IMREAD_COLOR)
        input_image_shape = img.shape
        img = cv2.resize(img, (self.model_input_shape[1],
                               self.model_input_shape[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255

        # Create intrinsics matrix
        intrinsics_mat = self.__create_intrinsics_matrix(
            input_image_shape, fov, focal_length)
        intrinsics_inv = np.linalg.inv(intrinsics_mat)

        # Predict depth
        img_batch = np.reshape(img, (1,) + img.shape)
        inverse_depth = self.depth_model.predict(img_batch)[0][0,:,:,0] 
        disparity_map = (self.min_disp + (self.max_disp - self.min_disp) 
                           * inverse_depth)
        predicted_median_depth = np.median(1 / disparity_map)
        depth = ((self.GT_DEPTH_SCALE / predicted_median_depth) 
                 * (1 / disparity_map))
        # Convert depth map to point cloud
        depth_tensor = K.variable(np.expand_dims(depth, 0))
        intrinsics_inv_tensor = K.variable(np.expand_dims(intrinsics_inv, 0))
        point_cloud = K.eval(get_cloud(depth_tensor, intrinsics_inv_tensor))
        point_cloud_flat = np.reshape(
            point_cloud, (point_cloud.shape[1] * point_cloud.shape[2], 3))

        # Predict segmentation masks
        masks_array = self.segmentator.infer_masks(input_image)
        print('[*] Found {} food object(s) '
              'in image.'.format(masks_array.shape[-1]))

        # Iterate over all predicted masks and estimate volumes
        estimated_volumes = []
        for k in range(masks_array.shape[-1]):
            # Apply mask to create object image and depth map
            object_mask = cv2.resize(masks_array[:,:,k], 
                                     (self.model_input_shape[1],
                                      self.model_input_shape[0]))
            object_img = (np.tile(np.expand_dims(object_mask, axis=-1),
                                     (1,1,3)) * img)
            object_depth = object_mask * depth
            # Get object/non-object points by filtering zero/non-zero 
            # depth pixels
            object_mask = (np.reshape(
                object_depth, (object_depth.shape[0] * object_depth.shape[1]))
                > 0)
            object_points = point_cloud_flat[object_mask, :]
            non_object_points = point_cloud_flat[np.logical_not(object_mask), :]

            # Estimate plate plane parameters
            plane_params = ransac_plane_estimation(object_points)
            # Filter outlier points
            object_points_filtered, sor_mask = sor_filter(
                object_points, 2, 0.7)
            # Transform object to match z-axis with plate normal
            translation, rotation_matrix = align_plane_with_axis(
                plane_params, np.array([0, 0, 1]))
            object_points_transformed = np.dot(
                object_points_filtered + translation, rotation_matrix.T)

            # Zero mean the transformed points and separate those 
            # over and under surface 
            object_points_transformed[:,2] -= np.mean(
                object_points_transformed[:,2])
            over_surface_points = (
                object_points_transformed[object_points_transformed[:,2] > 0])
            under_surface_points = (
                object_points_transformed[object_points_transformed[:,2] < 0])

            # Compute skewness of the each set's distances distribution 
            # as a measure of connectivity 
            # [more connected -> smaller distances -> higher skewness]
            over_surface_dist = pdist(over_surface_points, 'euclidean')
            over_surface_skew = skew(over_surface_dist)
            under_surface_dist = pdist(under_surface_points, 'euclidean')
            under_surface_skew = skew(under_surface_dist)

            # Use skewness measure to determine if the food surface is 
            # concave or convex and ...
            if over_surface_skew > under_surface_skew:
                print('[*] Concave food surface')
                distance_hist, bins = np.histogram(
                    object_points_transformed[:,2], bins=10)
                distance_density = distance_hist / np.sum(distance_hist)
                cum_density = np.cumsum(distance_density)
                z_adj_indx = next(x for x, val in enumerate(cum_density) 
                                  if val > 0.05)
                z_adj = (bins[z_adj_indx] + bins[z_adj_indx+1]) / 2
                object_points_transformed[:,2] += np.abs(z_adj)
            else:
                print('[*] Convex food surface')
                distance_hist, bins = np.histogram(
                    object_points_transformed[:,2], bins=10)
                distance_density = distance_hist / np.sum(distance_hist)
                cum_density = np.cumsum(distance_density)
                z_adj_indx = next(x for x, val in enumerate(cum_density) 
                                  if val > 0.95)
                z_adj = (bins[z_adj_indx] + bins[z_adj_indx+1]) / 2
                object_points_transformed[:,2] -= np.abs(z_adj)
                
            if plot_results:
                # Create object points from estimated plane
                plane_z = np.apply_along_axis(
                    lambda x: ((plane_params[0] + plane_params[1] * x[0]
                    + plane_params[2] * x[1]) * (-1) / plane_params[3]),
                    axis=1, arr=point_cloud_flat[:,:2])
                plane_points = np.concatenate(
                    (point_cloud_flat[:,:2], 
                    np.expand_dims(plane_z, axis=-1)), axis=-1)
                plane_points_transformed = np.dot(plane_points + translation, 
                                                  rotation_matrix.T)
                print('[*] Estimated plane parameters (w0,w1,w2,w3):',
                      plane_params)

                # Get the color values for the different sets of points
                colors_flat = (
                    np.reshape(img, (self.model_input_shape[0] 
                                     * self.model_input_shape[1], 3))
                    * 255)
                object_colors = colors_flat[object_mask, :]
                non_object_colors= colors_flat[np.logical_not(object_mask), :]
                object_colors_filtered = object_colors[sor_mask, :]

                # Create dataFrames for the different sets of points
                non_object_points_df = pd.DataFrame(
                    np.concatenate((non_object_points, non_object_colors), 
                                   axis=-1),
                    columns=['x','y','z','red','green','blue'])
                object_points_df = pd.DataFrame(
                    np.concatenate((object_points, object_colors), 
                    axis=-1), columns=['x','y','z','red','green','blue'])
                plane_points_df = pd.DataFrame(
                    plane_points, columns=['x','y','z'])
                object_points_transformed_df = pd.DataFrame(
                    np.concatenate((object_points_transformed, 
                                    object_colors_filtered), axis=-1),
                    columns=['x','y','z','red','green','blue'])
                plane_points_transformed_df = pd.DataFrame(
                    plane_points_transformed, columns=['x','y','z'])

                # Estimate volume
                estimated_volume, simplices = pc_to_volume(
                    object_points_transformed)
                print('[*] Estimated volume:', estimated_volume * 1000, 'L')

                # Plot input image and predicted segmentation mask/depth
                pretty_plotting([img, depth, object_img, object_depth], 
                                (2,2),
                                ['Input Image', 'Depth', 'Object Mask',
                                 'Object Depth'])
                plt.show()

                estimated_volumes.append(
                    (estimated_volume, object_points_df, non_object_points_df,
                     plane_points_df, object_points_transformed_df,
                     plane_points_transformed_df, simplices))
            else:
                # Estimate volume
                estimated_volume, _ = pc_to_volume(object_points_transformed)
                estimated_volumes.append(estimated_volume)

        return estimated_volumes

    def __create_intrinsics_matrix(self, input_image_shape, fov,
            focal_length):
        """Create intrinsics matrix from given camera parameters.
        Inputs:
            input_image_shape: Original input image shape.
            fov: Camera Field of View (in deg).
            focal_length: Camera focal length (in px).
        Returns:
            intrinsics_mat: Intrinsics matrix [3x3].
        """
        if fov is not None:
            F = input_image_shape[1] / (2 * np.tan((fov / 2) * np.pi / 180))
            print('[*] Creating intrinsics matrix from given FOV:', fov)
        elif focal_length is not None:
            F = focal_length
            print('[*] Creating intrinsics matrix from given focal length:',
                  focal_length)

        # Create intrinsics matrix
        x_scaling = int(self.model_input_shape[1]) / input_image_shape[1] 
        y_scaling = int(self.model_input_shape[0]) / input_image_shape[0] 
        intrinsics_mat = np.array(
            [[F * x_scaling, 0, (input_image_shape[1] / 2) * x_scaling], 
             [0, F * y_scaling, (input_image_shape[0] / 2) * y_scaling],
             [0, 0, 1]])
        return intrinsics_mat

    def __set_weights_trainable(self, model, trainable):
        """Sets model weights to trainable/non-trainable.
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

