import argparse
import numpy as np
import pandas as pd
import json
from keras.models import Model, model_from_json
import keras.preprocessing.image as pre
import matplotlib.pyplot as plt
from depth_estimation.networks import NetworkBuilder
from depth_estimation.custom_modules import *


class ModelTests:
    def __init__(self):
        """Initializes general parameters and loads models."""
        self.args = self.parse_args()
        # Load testing parameters 
        with open(self.args.config, 'r') as read_file:
            config = json.load(read_file)
            self.img_shape = tuple(config['img_size'])
            self.intrinsics_mat = np.array(config['intrinsics'])
            self.depth_range = config['depth_range']
            self.dataset = config['name']
        print('[*] Testing model on', self.dataset, 'dataset.')
        print('[*] Input image size:', self.img_shape)
        print('[*] Predicted depth range:', self.depth_range)
        
        # Create/Load model
        if self.args.model_architecture is None:
            # Network builder object 
            nets_builder = NetworkBuilder(
                self.img_shape, self.intrinsics_mat, self.depth_range)
            self.monovideo = nets_builder.create_monovideo()
            self.__set_weights_trainable(self.monovideo, False)
        else:
            objs = {'ProjectionLayer': ProjectionLayer, 
                    'ReflectionPadding2D': ReflectionPadding2D,
                    'InverseDepthNormalization': InverseDepthNormalization,
                    'AugmentationLayer': AugmentationLayer}
            with open(self.args.model_architecture, 'r') as read_file:
                model_architecture_json = json.load(read_file)
                self.monovideo = model_from_json(
                    model_architecture_json, custom_objects=objs)
        self.monovideo.load_weights(self.args.model_weights)

    def parse_args(self):
        """Parses command-line input arguments.

        Outputs:
            args: The arguments object.
        """
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Model testing script.')
        parser.add_argument('--test_outputs', action='store_true',
                            help='Test all model outputs.',
                            default=False)
        parser.add_argument('--infer_depth', action='store_true',
                            help='Infer depth of input images.',
                            default=False)
        parser.add_argument('--test_dataframe', type=str,
                            help='Test dataFrame file (.csv).',
                            default=None)
        parser.add_argument('--config', type=str, 
                            help='Dataset configuration file (.json).',
                            default=None)
        parser.add_argument('--model_architecture', type=str,
                            help='Model architecture file (.json).',
                            default=None)
        parser.add_argument('--model_weights', type=str,
                            help='Model weights file (.h5).',
                            default=None)
        parser.add_argument('--n_tests', type=int,
                            help='Number of tests.',
                            default=1)
        args = parser.parse_args()
        return args
    
    def infer_depth(self, n_tests):
        """Infer depth of input images using the depth estimation network.

        Inputs:
            n_tests: Number of inferences to perform.
        """
        # Slice loaded model to get depth model
        depth_net = self.monovideo.get_layer('depth_net')
        self.depth_model = Model(inputs=depth_net.inputs,
                                 outputs=depth_net.outputs,
                                 name='depth_model')
        # Create test data generator
        test_data_df = pd.read_csv(self.args.test_dataframe)
        self.test_data_gen = DataGenerator(
            test_data_df, self.img_shape[0], self.img_shape[1], 1, False)
        for i in range(n_tests):
            print('[-] Test Input [',i+1,'/',n_tests,']', sep='')
            test_data = self.test_data_gen.__getitem__(i)[0][0]
            outputs = self.depth_model.predict(test_data)
            # Predict and plot depth
            inverse_depth = outputs[0][0,:,:,0]
            depth = self.__inverse_depth_normalization(inverse_depth)
            self.__pretty_plotting([test_data[0], depth], (1,2),
                                   ['Input Frame', 'Predicted Depth'])
            plt.show()

    def test_outputs(self, n_tests):
        """Plots outputs of model on input images.

        Inputs:
            n_tests: Number of tests to perform.
        """
        # Create test data generator
        test_data_df = pd.read_csv(self.args.test_dataframe)
        self.test_data_gen = DataGenerator(
            test_data_df, self.img_shape[0], self.img_shape[1], 1, False)

        # Forward pass inputs
        for i in range(n_tests):
            print('[-] Test Input [',i+1,'/',n_tests,']', sep='')
            test_data = self.test_data_gen.__getitem__(i)[0]
            outputs = self.monovideo.predict(test_data)

            # Inputs
            inputs = [test_data[1][0], test_data[0][0], test_data[2][0]]
            input_titles = ['Previous Frame', 'Current Frame', 'Next Frame']
            self.__pretty_plotting(inputs, (1,3), input_titles)

            # Augmented inputs (if enabled during testing)
            aug_inputs = [outputs[1][0], outputs[0][0], outputs[2][0]]
            if np.sum(np.abs(np.concatenate(inputs, axis=-1) 
                             - np.concatenate(aug_inputs, axis=-1))) > 1e-5:
                aug_input_titles = ['Previous Frame (Aug.)',
                                    'Current Frame (Aug.)',
                                    'Next Frame (Aug.)']
                self.__pretty_plotting(aug_inputs, (1,3), aug_input_titles)

            # Reprojections
            reprojection_prev_1 = outputs[3][0]
            reprojection_next_1 = outputs[4][0]
            reprojection_prev_2 = outputs[5][0]
            reprojection_next_2 = outputs[6][0]
            reprojection_prev_3 = outputs[7][0]
            reprojection_next_3 = outputs[8][0]
            reprojection_prev_4 = outputs[9][0]
            reprojection_next_4 = outputs[10][0]

            reprojections = [reprojection_prev_1, reprojection_next_1,
                             reprojection_prev_2, reprojection_next_2,
                             reprojection_prev_3, reprojection_next_3,
                             reprojection_prev_4, reprojection_next_4]
            reprojection_titles = [
                'Reprojection Prev. (S1)', 'Reprojection Next (S1)',
                'Reprojection Prev. (S2)', 'Reprojection Next (S2)',
                'Reprojection Prev. (S3)', 'Reprojection Next (S3)',
                'Reprojection Prev. (S4)', 'Reprojection Next (S4)']
            self.__pretty_plotting(reprojections, (4,2), reprojection_titles)

            # Inverse depths
            depth_1 = outputs[19][0,:,:,0]
            depth_2 = outputs[20][0,:,:,0]
            depth_3 = outputs[21][0,:,:,0]
            depth_4 = outputs[22][0,:,:,0]
            depths = [depth_1, depth_2, depth_3, depth_4]
            depth_titles = ['Inferred Depth (S1)', 'Inferred Depth (S2)',
                            'Inferred Depth (S3)', 'Inferred Depth (S4)']
            self.__pretty_plotting(depths, (2,2), depth_titles)
            plt.show()

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

    def __inverse_depth_normalization(self, x):
        min_disp = 1 / self.depth_range[1]
        max_disp = 1 / self.depth_range[0]
        normalized_disp = (min_disp + (max_disp - min_disp) * x)
        depth_map = 1 / normalized_disp
        return depth_map

    def __pretty_plotting(self, imgs, tiling, titles):
        """Plots images in a pretty fashion.

        Inputs:
            imgs: List of images to plot.
            tiling: Subplot tiling tuple (rows,cols).
            titles: List of subplot titles.
        """
        n_plots = len(imgs)
        rows = str(tiling[0])
        cols = str(tiling[1])
        plt.figure()
        for r in range(tiling[0] * tiling[1]):
            plt.subplot(rows + cols + str(r + 1))
            plt.title(titles[r])
            plt.imshow(imgs[r])



if __name__ == '__main__':
    model_tests = ModelTests()

    if model_tests.args.test_outputs == True:
        model_tests.test_outputs(model_tests.args.n_tests)
    elif model_tests.args.infer_depth == True:
        model_tests.infer_depth(model_tests.args.n_tests)
    else:
        print('[!] Unknown operation, use -h flag for help.')

