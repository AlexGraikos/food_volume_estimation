import argparse
import numpy as np
import pandas as pd
from keras.models import model_from_json
import keras.preprocessing.image as pre
import matplotlib.pyplot as plt
import json

import custom_modules

class ModelTests:
    def __init__(self):
        """"
        Initializes general parameters and loads models
        """
        self.args = self.parse_args()
        # Load model
        objs = {'ProjectionLayer': custom_modules.ProjectionLayer, 
                'ReflectionPadding2D': custom_modules.ReflectionPadding2D}
        with open(self.args.model_file, 'r') as read_file:
            model_json = json.load(read_file)
            self.test_model = model_from_json(model_json, custom_objects=objs)
        self.test_model.load_weights(self.args.model_weights)
        self.img_shape = self.test_model.inputs[0].shape[1:]
        # Training data generator
        test_data_df = pd.read_csv(self.args.test_dataframe)
        self.testDatagen = self.test_datagen(test_data_df, self.img_shape[0],
            self.img_shape[1], self.args.n_tests)


    def parse_args(self):
        """
        Parses command-line input arguments
            Outputs:
                args: The arguments object
        """
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Model testing script.')
        parser.add_argument('--test_generator', action='store_true', help='Test all generator outputs.', default=False)
        parser.add_argument('--infer_depth', action='store_true', help='Infer depth from input images.', default=False)
        parser.add_argument('--test_dataframe', type=str, help='File containing the test dataFrame.', default=None)
        parser.add_argument('--n_tests', type=int, help='Number of tests.', default=1)
        parser.add_argument('--model_file', type=str, help='Model architecture file (.json).', default=None)
        parser.add_argument('--model_weights', type=str, help='Model weights file (.h5).', default=None)
        args = parser.parse_args()
        return args
 

    def infer_depth(self, n_tests):
        """
        Gets model reprojections and plots them
            Inputs:
                n_tests: Number of tests to perform
        """
        # Predict outputs 
        test_data = self.testDatagen.__next__()
        results = self.test_model.predict(test_data)
        for i in range(n_tests):
            print('[-] Test [',i+1,'/',n_tests,']',sep='')

            depth1 = custom_modules.inverseDepthNormalization(results[8][i,:,:,0])
            depth2 = custom_modules.inverseDepthNormalization(results[9][i,:,:,0])
            depth3 = custom_modules.inverseDepthNormalization(results[10][i,:,:,0])
            depth4 = custom_modules.inverseDepthNormalization(results[11][i,:,:,0])

            # Inputs
            plt.figure()
            plt.subplot(131)
            plt.title('Previous frame')
            plt.imshow(test_data[1][i])

            plt.subplot(132)
            plt.title('Current frame')
            plt.imshow(test_data[0][i])

            plt.subplot(133)
            plt.title('Next frame')
            plt.imshow(test_data[2][i])

            # Inverse depths
            plt.figure()
            plt.subplot(221)
            plt.title('Inferred depth (scale 1)')
            plt.imshow(depth1)
            
            plt.subplot(222)
            plt.title('Inferred depth (scale 2)')
            plt.imshow(depth2)
            
            plt.subplot(223)
            plt.title('Inferred depth (scale 3)')
            plt.imshow(depth3)
            
            plt.subplot(224)
            plt.title('Inferred depth (scale 4)')
            plt.imshow(depth4)
            plt.show()


    def test_generator(self, n_tests):
        """
        Infers depth of input image and plots
            Inputs:
                n_tests: Number of tests to perform
        """
        # Infer depth
        test_data = self.testDatagen.__next__()
        results = self.test_model.predict(test_data)
        for i in range(n_tests):
            print('[-] Test [',i+1,'/',n_tests,']',sep='')

            reprojection1Prev = results[0][i]
            reprojection1Next = results[1][i]
            reprojection2Prev = results[2][i]
            reprojection2Next = results[3][i]
            reprojection3Prev = results[4][i]
            reprojection3Next = results[5][i]
            reprojection4Prev = results[6][i]
            reprojection4Next = results[7][i]
            depth1 = custom_modules.inverseDepthNormalization(results[8][i,:,:,0])
            depth2 = custom_modules.inverseDepthNormalization(results[9][i,:,:,0])
            depth3 = custom_modules.inverseDepthNormalization(results[10][i,:,:,0])
            depth4 = custom_modules.inverseDepthNormalization(results[11][i,:,:,0])

            # Inputs
            plt.figure()
            plt.subplot(131)
            plt.title('Previous frame')
            plt.imshow(test_data[1][i])
            plt.subplot(132)
            plt.title('Current frame')
            plt.imshow(test_data[0][i])
            plt.subplot(133)
            plt.title('Next frame')
            plt.imshow(test_data[2][i])

            # Reprojections
            plt.figure()
            plt.subplot(421)
            plt.title('Reprojection Prev (scale 1)')
            plt.imshow(reprojection1Prev)
            plt.subplot(422)
            plt.title('Reprojection Next (scale 1)')
            plt.imshow(reprojection1Next)
            plt.subplot(423)
            plt.title('Reprojection Prev (scale 2)')
            plt.imshow(reprojection2Prev)
            plt.subplot(424)
            plt.title('Reprojection Next (scale 2)')
            plt.imshow(reprojection2Next)
            plt.subplot(425)
            plt.title('Reprojection Prev (scale 3)')
            plt.imshow(reprojection3Prev)
            plt.subplot(426)
            plt.title('Reprojection Next (scale 3)')
            plt.imshow(reprojection3Next)
            plt.subplot(427)
            plt.title('Reprojection Prev (scale 4)')
            plt.imshow(reprojection4Prev)
            plt.subplot(428)
            plt.title('Reprojection Next (scale 4)')
            plt.imshow(reprojection4Next)

            # Inverse depths
            plt.figure()
            plt.subplot(221)
            plt.title('Inferred depth (scale 1)')
            plt.imshow(depth1)
            plt.subplot(222)
            plt.title('Inferred depth (scale 2)')
            plt.imshow(depth2)
            plt.subplot(223)
            plt.title('Inferred depth (scale 3)')
            plt.imshow(depth3)
            plt.subplot(224)
            plt.title('Inferred depth (scale 4)')
            plt.imshow(depth4)
            plt.show()


    def test_datagen(self, test_data_df, height, width, n_tests):
        """
        Creates test data generator for the model tests
            Inputs:
                test_data_df: The dataframe containing the paths to frame triplets
                height: Input image height
                width: Input image width
                n_tests: Generated batch sizes
            Outputs:
                (inputs) tuple for tests
        """
        # Image preprocessor
        datagen = pre.ImageDataGenerator(
                rescale = 1./255,
                channel_shift_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest')

        # Frame generators
        seed = int(np.random.rand(1,1)*1000) # Using same seed to ensure temporal continuity
        curr_generator = datagen.flow_from_dataframe(test_data_df, directory=None,
            x_col='curr_frame', target_size=(height,width), batch_size=n_tests,
            interpolation='bilinear', class_mode=None, seed=seed)
        prev_generator = datagen.flow_from_dataframe(test_data_df, directory=None,
            x_col='prev_frame', target_size=(height,width), batch_size=n_tests,
            interpolation='bilinear', class_mode=None, seed=seed)
        next_generator = datagen.flow_from_dataframe(test_data_df, directory=None,
            x_col='next_frame', target_size=(height,width), batch_size=n_tests,
            interpolation='bilinear', class_mode=None, seed=seed)
    
        while True:
            curr_frame = curr_generator.__next__()
            prev_frame = prev_generator.__next__()
            next_frame = next_generator.__next__()

            yield ([curr_frame, prev_frame, next_frame])



if __name__ == '__main__':
    model_tests = ModelTests()

    if model_tests.args.test_generator == True:
        model_tests.test_generator(model_tests.args.n_tests)
    elif model_tests.args.infer_depth== True:
        model_tests.infer_depth(model_tests.args.n_tests)
    else:
        print('[!] Unknown operation argument, use -h flag for help.')

