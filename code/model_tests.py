import argparse
import numpy as np
import pandas as pd
from keras.models import model_from_json
import keras.preprocessing.image as pre
import matplotlib.pyplot as plt
import json
from custom_modules import *


class ModelTests:
    def __init__(self):
        """"
        Initializes general parameters and loads models.
        """
        self.args = self.parse_args()
        # Load model architecture with custom model objects
        objs = {'ProjectionLayer': ProjectionLayer, 
                'ReflectionPadding2D': ReflectionPadding2D,
                'InverseDepthNormalization': InverseDepthNormalization,
                'AugmentationLayer': AugmentationLayer}
        with open(self.args.model_file, 'r') as read_file:
            model_architecture_json = json.load(read_file)
            self.test_model = model_from_json(model_architecture_json,
                                              custom_objects=objs)
        # Load weights
        self.test_model.load_weights(self.args.model_weights)
        self.img_shape = self.test_model.inputs[0].shape[1:]
        # Create test data generator
        test_data_df = pd.read_csv(self.args.test_dataframe)
        self.test_data_gen = self.create_test_data_gen(
            test_data_df, self.img_shape[0], self.img_shape[1])


    def parse_args(self):
        """
        Parses command-line input arguments.
            Outputs:
                args: The arguments object.
        """
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Model testing script.')
        parser.add_argument('--test_outputs', action='store_true',
                            help='Test all model outputs.',
                            default=False)
        parser.add_argument('--test_dataframe', type=str,
                            help='File containing the test dataFrame.',
                            default=None)
        parser.add_argument('--model_file', type=str,
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
    

    def test_outputs(self, n_tests):
        """
        Plots outputs of model on input images.
            Inputs:
                n_tests: Number of tests to perform.
        """
        # Infer depth
        for i in range(n_tests):
            print('[-] Test Input [',i+1,'/',n_tests,']',sep='')
            test_data = self.test_data_gen.__next__()
            outputs = self.test_model.predict(test_data)

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


    def create_test_data_gen(self, test_data_df, height, width):
        """
        Creates test data generator for the model tests.
            Inputs:
                test_data_df: The dataframe containing the paths 
                              to frame triplets.
                height: Input image height.
                width: Input image width.
                n_tests: Generated batch sizes.
            Outputs:
                (inputs) tuple for tests.
        """
        # Image preprocessor
        datagen = pre.ImageDataGenerator(rescale=1/255, fill_mode='nearest')

        # Frame generators - use same seed to ensure continuity
        seed = int(np.random.rand(1,1)*1000)
        curr_generator = datagen.flow_from_dataframe(
            test_data_df, directory=None, x_col='curr_frame',
            target_size=(height,width), batch_size=1,
            interpolation='bilinear', class_mode=None, seed=seed)
        prev_generator = datagen.flow_from_dataframe(
            test_data_df, directory=None, x_col='prev_frame',
            target_size=(height,width), batch_size=1,
            interpolation='bilinear', class_mode=None, seed=seed)
        next_generator = datagen.flow_from_dataframe(
            test_data_df, directory=None, x_col='next_frame',
            target_size=(height,width), batch_size=1,
            interpolation='bilinear', class_mode=None, seed=seed)
    
        while True:
            curr_frame = curr_generator.__next__()
            prev_frame = prev_generator.__next__()
            next_frame = next_generator.__next__()

            yield ([curr_frame, prev_frame, next_frame])


    def __pretty_plotting(self, imgs, tiling, titles):
        """
        Plots images in a pretty fashion.
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
    elif model_tests.args.create_pc == True:
        model_tests.create_point_cloud(model_tests.args.input_image)
    else:
        print('[!] Unknown operation, use -h flag for help.')

