import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import json
from scipy.spatial.distance import pdist
from scipy.stats import skew
from keras.models import Model, model_from_json
import keras.backend as K
from food_segmentation.food_segmentator import FoodSegmentator
from depth_estimation.custom_modules import *
from depth_estimation.project import *
from volume_estimation.point_cloud_utils import *
from ellipse_detection.ellipse_detector import EllipseDetector
import matplotlib.pyplot as plt


class VolumeEstimator():
    """Volume estimator object."""
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
            self.model_input_shape = (
                self.monovideo.inputs[0].shape.as_list()[1:])
            depth_net = self.monovideo.get_layer('depth_net')
            self.depth_model = Model(inputs=depth_net.inputs,
                                     outputs=depth_net.outputs,
                                     name='depth_model')
            print('[*] Loaded depth estimation model.')
            # Depth model configuration
            self.min_disp = 1 / self.args.max_depth
            self.max_disp = 1 / self.args.min_depth
            self.gt_depth_scale = self.args.gt_depth_scale

            # Create segmentator object
            self.segmentator = FoodSegmentator(self.args.segmentation_weights)

            # Plate adjustment relaxation parameter
            self.relax_param = self.args.relaxation_param

    def __parse_args(self):
        """Parse command-line input arguments.

        Returns:
            args: The arguments object.
        """
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description='Estimate food volume in input image.')
        parser.add_argument('--input_images', type=str, nargs='+',
                            help='Paths to input images.',
                            metavar='/path/to/image1 /path/to/image2 ...',
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
        parser.add_argument('--plate_diameter_prior', type=float,
                            help=('Expected plate diameter (in m) .'
                                  + ' or 0 to ignore plate scaling'),
                            metavar='<plate_diameter_prior>',
                            default=0.3)
        parser.add_argument('--gt_depth_scale', type=float,
                            help='Ground truth depth rescaling factor.',
                            metavar='<gt_depth_scale>',
                            default=0.35)
        parser.add_argument('--min_depth', type=float,
                            help='Minimum depth value.',
                            metavar='<min_depth>',
                            default=0.01)
        parser.add_argument('--max_depth', type=float,
                            help='Maximum depth value.',
                            metavar='<max_depth>',
                            default=10)
        parser.add_argument('--relaxation_param', type=float,
                            help='Plate adjustment relaxation parameter.',
                            metavar='<relaxation_param>',
                            default=0.01)
        parser.add_argument('--plot_results', action='store_true',
                            help='Plot volume estimation results.',
                            default=False)
        parser.add_argument('--results_file', type=str,
                            help='File to save results at (.csv).',
                            metavar='/path/to/results.csv',
                            default=None)
        args = parser.parse_args()
        
        assert (args.fov is not None) or (args.focal_length is not None), (
            '[!] Either the camera FoV or Focal Length must be provided')

        return args

    def estimate_volume(self, input_image, fov, focal_length, 
            plate_diameter_prior, plot_results):
        """Volume estimation procedure.

        Inputs:
            input_image: Path to input image.
            fov: Camera Field of View.
            focal_length: Camera Focal length.
            plate_diameter_prior: Expected plate diameter.
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
        depth = 1 / disparity_map
        # Convert depth map to point cloud
        depth_tensor = K.variable(np.expand_dims(depth, 0))
        intrinsics_inv_tensor = K.variable(np.expand_dims(intrinsics_inv, 0))
        point_cloud = K.eval(get_cloud(depth_tensor, intrinsics_inv_tensor))
        point_cloud_flat = np.reshape(
            point_cloud, (point_cloud.shape[1] * point_cloud.shape[2], 3))

        # Find the ellipse parameterss (cx, cy, a, b, theta) that 
        # describe the plate contour
        ellipse_scale = 2.5
        ellipse_detector = EllipseDetector(
            (ellipse_scale * self.model_input_shape[0],
             ellipse_scale * self.model_input_shape[1]))
        ellipse_params = ellipse_detector.detect(input_image)
        ellipse_params_scaled = tuple(
            [x / ellipse_scale for x in ellipse_params[:-1]]
            + [ellipse_params[-1]])

        # Scale depth map
        if (any(x != 0 for x in ellipse_params_scaled) and
                plate_diameter_prior != 0):
            print('[*] Ellipse parameters:', ellipse_params_scaled)
            # Find the scaling factor to match prior 
            # and measured plate diameters
            plate_point_1 = [int(ellipse_params_scaled[2] 
                             * np.sin(ellipse_params_scaled[4]) 
                             + ellipse_params_scaled[1]), 
                             int(ellipse_params_scaled[2] 
                             * np.cos(ellipse_params_scaled[4]) 
                             + ellipse_params_scaled[0])]
            plate_point_2 = [int(-ellipse_params_scaled[2] 
                             * np.sin(ellipse_params_scaled[4]) 
                             + ellipse_params_scaled[1]),
                             int(-ellipse_params_scaled[2] 
                             * np.cos(ellipse_params_scaled[4]) 
                             + ellipse_params_scaled[0])]
            plate_point_1_3d = point_cloud[0, plate_point_1[0], 
                                           plate_point_1[1], :]
            plate_point_2_3d = point_cloud[0, plate_point_2[0], 
                                           plate_point_2[1], :]
            plate_diameter = np.linalg.norm(plate_point_1_3d 
                                            - plate_point_2_3d)
            scaling = plate_diameter_prior / plate_diameter
        else:
            # Use the median ground truth depth scaling
            print('[*] No ellipse found. Scaling with expected median depth.')
            predicted_median_depth = np.median(1 / disparity_map)
            scaling = self.gt_depth_scale / predicted_median_depth
        depth = scaling * depth
        point_cloud = scaling * point_cloud
        point_cloud_flat = scaling * point_cloud_flat

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
            non_object_points = point_cloud_flat[
                np.logical_not(object_mask), :]

            # Filter outlier points
            object_points_filtered, sor_mask = sor_filter(
                object_points, 2, 0.7)
            # Estimate base plane parameters
            plane_params = pca_plane_estimation(object_points_filtered)
            # Transform object to match z-axis with plane normal
            translation, rotation_matrix = align_plane_with_axis(
                plane_params, np.array([0, 0, 1]))
            object_points_transformed = np.dot(
                object_points_filtered + translation, rotation_matrix.T)

            # Adjust object on base plane
            height_sorted_indices = np.argsort(object_points_transformed[:,2])
            adjustment_index = height_sorted_indices[
                int(object_points_transformed.shape[0] * self.relax_param)]
            object_points_transformed[:,2] += np.abs(
                object_points_transformed[adjustment_index, 2])
             
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

                # Outline the detected plate contour and major axis vertices 
                plate_contour = np.copy(img)
                if (any(x != 0 for x in ellipse_params_scaled) and
                        plate_diameter_prior != 0):
                    ellipse_color = (68 / 255, 1 / 255, 84 / 255)
                    vertex_color = (253 / 255, 231 / 255, 37 / 255)
                    cv2.ellipse(plate_contour,
                                (int(ellipse_params_scaled[0]),
                                 int(ellipse_params_scaled[1])), 
                                (int(ellipse_params_scaled[2]),
                                 int(ellipse_params_scaled[3])),
                                ellipse_params_scaled[4] * 180 / np.pi, 
                                0, 360, ellipse_color, 2)
                    cv2.circle(plate_contour,
                               (int(plate_point_1[1]), int(plate_point_1[0])),
                               2, vertex_color, -1)
                    cv2.circle(plate_contour,
                               (int(plate_point_2[1]), int(plate_point_2[0])),
                               2, vertex_color, -1)

                # Estimate volume for points above the plane
                volume_points = object_points_transformed[
                    object_points_transformed[:,2] > 0]
                estimated_volume, simplices = pc_to_volume(volume_points)
                print('[*] Estimated volume:', estimated_volume * 1000, 'L')

                # Plot input image and predicted segmentation mask/depth
                pretty_plotting([img, plate_contour, depth, object_img], 
                                (2,2),
                                ['Input Image', 'Plate Contour', 'Depth', 
                                 'Object Mask'],
                                'Estimated Volume: {:.3f} L'.format(
                                estimated_volume * 1000.0))
                plt.show()

                estimated_volumes.append(
                    (estimated_volume, object_points_df, non_object_points_df,
                     plane_points_df, object_points_transformed_df,
                     plane_points_transformed_df, simplices))
            else:
                # Estimate volume for points above the plane
                volume_points = object_points_transformed[
                    object_points_transformed[:,2] > 0]
                estimated_volume, _ = pc_to_volume(volume_points)
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

    # Iterate over input images, estimate volumes and store results
    results = {'image_path': [], 'volumes': []}
    for input_image in estimator.args.input_images:
        print('[*] Input:', input_image)

        volumes = estimator.estimate_volume(
            input_image, estimator.args.fov, estimator.args.focal_length,
            estimator.args.plate_diameter_prior, estimator.args.plot_results)

        results['image_path'].append(input_image)
        if estimator.args.plot_results:
            results['volumes'].append([x[0] for x in volumes])
        else:
            results['volumes'].append(volumes)

    if estimator.args.results_file is not None:
        # Save results in CSV format
        volumes_df = pd.DataFrame(data=results)
        volumes_df.to_csv(estimator.args.results_file, index=False)

