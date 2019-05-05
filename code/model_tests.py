import argparse
import numpy as np
import pandas as pd
from keras.models import load_model
import keras.preprocessing.image as pre
from extras import *
import matplotlib.pyplot as plt

class ModelTests:

    def __init__(self):
        """"
        Initializes general parameters and loads models
        """
        self.args = self.parse_args()
        # Model parameters
        # Should customize for each separate model
        custom_objects = {'ProjectionLayer': ProjectionLayer}
        self.test_model= load_model(self.args.model_file, custom_objects=custom_objects)
        self.img_shape = self.test_model.inputs[0].shape[1:]

        # ?????
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # ?????


    def parse_args(self):
        """
        Parses command-line input arguments
            Outputs:
                args: The arguments object
        """
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Model testing script.')
        parser.add_argument('--infer_depth', action='store_true', help='Infer depth from input images.', default=False)
        parser.add_argument('--test_generator', action='store_true', help='Test generator model.', default=False)
        parser.add_argument('--test_dataframe', type=str, help='File containing the test dataFrame.', default=None)
        parser.add_argument('--n_tests', type=int, help='Number of tests.', default=1)
        parser.add_argument('--model_file', type=str, help='File that model is saved at (.h5).', default='test_model.h5')
        args = parser.parse_args()
        return args

 
    def test_generator_model(self, test_df_file, n_tests):
        """
        Tests generator model
            Inputs:
                test_df_file: Test data dataFrame file (.h5, 'df' key)
                n_tests: Number of tests to perform
        """
        # Training data generators
        test_data_df = pd.read_hdf(test_df_file, 'df')
        generator_tests_datagen = self.test_datagen(test_data_df, self.img_shape[0],
            self.img_shape[1], n_tests)

        # Test model
        test_data = generator_tests_datagen.__next__()
        # Predict outputs 
        outputs = self.test_model.predict(test_data)
        for i in range(n_tests):
            print('[-] Test [',i+1,'/',n_tests,']',sep='')

            # Print results
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

            plt.figure()
            plt.subplot(221)
            plt.title('Reprojection (scale 1 - prev)')
            plt.imshow(outputs[0][i])

            plt.subplot(222)
            plt.title('Reprojection (scale 1 - next)')
            plt.imshow(outputs[1][i])

            '''
            plt.subplot(223)
            plt.title('Reprojection (scale 2 - prev)')
            plt.imshow(outputs[2][i])

            plt.subplot(224)
            plt.title('Reprojection (scale 2 - next)')
            plt.imshow(outputs[3][i])

            plt.figure()
            plt.subplot(221)
            plt.title('Reprojection (scale 3 - prev)')
            plt.imshow(outputs[4][i])

            plt.subplot(222)
            plt.title('Reprojection (scale 3 - next)')
            plt.imshow(outputs[5][i])

            plt.subplot(223)
            plt.title('Reprojection (scale 4 - prev)')
            plt.imshow(outputs[6][i])

            plt.subplot(224)
            plt.title('Reprojection (scale 4 - next)')
            plt.imshow(outputs[7][i])
            '''
            plt.show()


    def test_depth_model(self, test_df_file, n_tests):
        """
        Tests depth inference model
            Inputs:
                test_df_file: Test data dataFrame file (.h5, 'df' key)
                n_tests: Number of tests to perform
        """
        # Training data generators
        test_data_df = pd.read_hdf(test_df_file, 'df')
        depth_tests_datagen = self.test_datagen(test_data_df, self.img_shape[0],
            self.img_shape[1], n_tests)

        # Test model
        test_data = depth_tests_datagen.__next__()
        # Infer depth
        inverse_depth = self.test_model.predict(test_data[0])
        for i in range(n_tests):
            print('[-] Test [',i+1,'/',n_tests,']',sep='')

            # Print results
            plt.figure(i)
            plt.subplot(321)
            plt.title('Input image')
            plt.imshow(test_data[0][i])

            plt.subplot(323)
            plt.title('Inferred depth (scale 1)')
            plt.imshow(1/inverse_depth[i,:,:,0])
            
            '''
            plt.subplot(324)
            plt.title('Inferred depth (scale 2)')
            plt.imshow(inverse_depth[1][i,:,:,0])
            
            plt.subplot(325)
            plt.title('Inferred depth (scale 3)')
            plt.imshow(inverse_depth[2][i,:,:,0])
            
            plt.subplot(326)
            plt.title('Inferred depth (scale 4)')
            plt.imshow(inverse_depth[3][i,:,:,0])
            '''
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

    if model_tests.args.infer_depth == True:
        model_tests.test_depth_model(model_tests.args.test_dataframe, model_tests.args.n_tests)
    elif model_tests.args.test_generator == True:
        model_tests.test_generator_model(model_tests.args.test_dataframe, model_tests.args.n_tests)
    else:
        print('[!] Unknown operation argument, use -h flag for help.')

