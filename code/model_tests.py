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
                'InverseDepthNormalization': InverseDepthNormalization}
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
            test_data_df, self.img_shape[0], self.img_shape[1],
            self.args.n_tests)


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
        parser.add_argument('--infer_depth', action='store_true',
                            help='Infer depth from input images.',
                            default=False)
        parser.add_argument('--test_dataframe', type=str,
                            help='File containing the test dataFrame.',
                            default=None)
        parser.add_argument('--n_tests', type=int,
                            help='Number of tests.',
                            default=1)
        parser.add_argument('--model_file', type=str,
                            help='Model architecture file (.json).',
                            default=None)
        parser.add_argument('--model_weights', type=str,
                            help='Model weights file (.h5).',
                            default=None)
        args = parser.parse_args()
        return args
 

    def infer_depth(self, n_tests):
        """
        Infers depth from input image.
            Inputs:
                n_tests: Number of tests to perform.
        """
        # Predict outputs 
        test_data = self.test_data_gen.__next__()
        outputs = self.test_model.predict(test_data)
        for i in range(n_tests):
            print('[-] Test Input [',i+1,'/',n_tests,']',sep='')

            # Inputs
            inputs = [test_data[1][i], test_data[0][i], test_data[2][i]]
            input_titles = ['Previous Frame', 'Current Frame', 'Next Frame']
            self.__pretty_plotting(inputs, (1,3), input_titles)

            # Inverse depths
            depth_1 = self.__normalize_inverse_depth(
                outputs[8][i,:,:,0], 0.1, 10)
            depth_2 = self.__normalize_inverse_depth(
                outputs[9][i,:,:,0], 0.1, 10, upsample=2)
            depth_3 = self.__normalize_inverse_depth(
                outputs[10][i,:,:,0], 0.1, 10, upsample=4)
            depth_4 = self.__normalize_inverse_depth(
                outputs[11][i,:,:,0], 0.1, 10, upsample=8)
            depths = [depth_1, depth_2, depth_3, depth_4]
            depth_titles = ['Inferred Depth (S1)', 'Inferred Depth (S2)',
                            'Inferred Depth (S3)', 'Inferred Depth (S4)']
            self.__pretty_plotting(depths, (2,2), depth_titles)
            plt.show()


    def test_outputs(self, n_tests):
        """
        Plots outputs of model on input images.
            Inputs:
                n_tests: Number of tests to perform.
        """
        # Infer depth
        test_data = self.test_data_gen.__next__()
        outputs = self.test_model.predict(test_data)
        for i in range(n_tests):
            print('[-] Test Input [',i+1,'/',n_tests,']',sep='')

            # Inputs
            inputs = [test_data[1][i], test_data[0][i], test_data[2][i]]
            input_titles = ['Previous Frame', 'Current Frame', 'Next Frame']
            self.__pretty_plotting(inputs, (1,3), input_titles)

            # Reprojections
            reprojection_prev_1 = outputs[0][i]
            reprojection_next_1 = outputs[1][i]
            reprojection_prev_2 = outputs[2][i]
            reprojection_next_2 = outputs[3][i]
            reprojection_prev_3 = outputs[4][i]
            reprojection_next_3 = outputs[5][i]
            reprojection_prev_4 = outputs[6][i]
            reprojection_next_4 = outputs[7][i]

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
            depth_1 = self.__normalize_inverse_depth(
                outputs[8][i,:,:,0], 0.1, 10)
            depth_2 = self.__normalize_inverse_depth(
                outputs[9][i,:,:,0], 0.1, 10, upsample=2)
            depth_3 = self.__normalize_inverse_depth(
                outputs[10][i,:,:,0], 0.1, 10, upsample=4)
            depth_4 = self.__normalize_inverse_depth(
                outputs[11][i,:,:,0], 0.1, 10, upsample=8)
            depths = [depth_1, depth_2, depth_3, depth_4]
            depth_titles = ['Inferred Depth (S1)', 'Inferred Depth (S2)',
                            'Inferred Depth (S3)', 'Inferred Depth (S4)']
            self.__pretty_plotting(depths, (2,2), depth_titles)
            plt.show()


    def create_test_data_gen(self, test_data_df, height, width, n_tests):
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
        datagen = pre.ImageDataGenerator(
            rescale=1/255,
            channel_shift_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest')

        # Frame generators - use same seed to ensure continuity
        seed = int(np.random.rand(1,1)*1000)
        curr_generator = datagen.flow_from_dataframe(
            test_data_df, directory=None, x_col='curr_frame',
            target_size=(height,width), batch_size=n_tests,
            interpolation='bilinear', class_mode=None, seed=seed)
        prev_generator = datagen.flow_from_dataframe(
            test_data_df, directory=None, x_col='prev_frame',
            target_size=(height,width), batch_size=n_tests,
            interpolation='bilinear', class_mode=None, seed=seed)
        next_generator = datagen.flow_from_dataframe(
            test_data_df, directory=None, x_col='next_frame',
            target_size=(height,width), batch_size=n_tests,
            interpolation='bilinear', class_mode=None, seed=seed)
    
        while True:
            curr_frame = curr_generator.__next__()
            prev_frame = prev_generator.__next__()
            next_frame = next_generator.__next__()

            yield ([curr_frame, prev_frame, next_frame])


    def __normalize_inverse_depth(self, disp, min_depth, max_depth, 
            upsample=1):
        """
        Upsamples input disparity map and returns normalized depth with 
        given min and max values.
            Inputs:
                disp: Input disparity map
                min_depth: Minimum depth value.
                max_depth: Maximum depth value.
                upsample: Upsampling rate.
            Outputs:
                depth_map: Produced depth map.
        """
        if upsample > 1:
            disp = np.repeat(np.repeat(disp, upsample, axis=0),
                             upsample, axis=1)
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        normalized_disp = min_disp + (max_disp - min_disp) * disp
        depth_map = 1 / normalized_disp
        return depth_map 


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
    elif model_tests.args.infer_depth== True:
        model_tests.infer_depth(model_tests.args.n_tests)
    else:
        print('[!] Unknown operation, use -h flag for help.')

