import argparse
import numpy as np
import pandas as pd
import os
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D, ZeroPadding2D, GlobalAveragePooling2D, LeakyReLU, Dense, Flatten
from keras.layers import Input, Concatenate, Lambda
from keras.optimizers import Adam
from keras import callbacks
import keras.preprocessing.image as pre
from classification_models.resnet import ResNet18
from extras import *

class MonocularAdversarialModel:

    def __init__(self):
        """"
        Initializes general parameters and creates models
        """
        self.args = self.parse_args()
        # Model parameters
        self.img_shape = (self.args.img_height, self.args.img_width, 3)
        self.model_name = self.args.model_name
        self.omega = self.args.omega
        # Intrinsic parameters
        self.latent_depth_features_shape = None # Depth encoder output shape
        self.skip_connection_shapes = None # Encoder-Decoder skip connection shapes


    def parse_args(self):
        """
        Parses command-line input arguments
            Outputs:
                args: The arguments object
        """
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Model training script.')
        parser.add_argument('--train', action='store_true', help='Train model.', default=False)
        parser.add_argument('--train_dataframe', type=str, help='File containing the training dataFrame.', default=None)
        parser.add_argument('--batch_size', type=int, help='Training batch size.', default=8)
        parser.add_argument('--training_epochs', type=int, help='Model training epochs.', default=10)
        parser.add_argument('--omega', type=float, help='Discriminator noise coefficient.', default=0.9)
        parser.add_argument('--model_name', type=str, help='Name to use for saving model.', default='monovideo')
        parser.add_argument('--img_width', type=int, help='Input image width.', default=224)
        parser.add_argument('--img_height', type=int, help='Input image height.', default=128)
        args = parser.parse_args()
        return args


    def train(self, train_df_file, batch_size, training_epochs):
        """
        Train model
            Inputs:
                train_df_file: Training data dataFrame file (.h5, 'df' key)
                batch_size: Training batch size
                training_epochs: Number of training epochs
        """

        # Learning rate halving callback function
        def halve_lr(epoch, curr_lr):
            # Epochs are zero-indexed
            if (epoch < 10): # Halves at epoch 11
                return curr_lr
            else:
                if (epoch % 10 == 0): # Halves every 10 epochs
                    print('[*] Halving learning rate (=',curr_lr,' -> ',(curr_lr / 2.0),')',sep='')
                    return curr_lr / 2.0
                else:
                    return curr_lr
        callback = [callbacks.LearningRateScheduler(schedule=halve_lr, verbose=0)]
        
        # Training data generators
        training_data_df = pd.read_hdf(train_df_file, 'df')
        discriminatorDatagen = self.discriminator_train_datagen(training_data_df, self.img_shape[0],
            self.img_shape[1], batch_size)
        generatorDatagen = self.generator_train_datagen(training_data_df, self.img_shape[0], self.img_shape[1],
            batch_size)

        # Keras bug - model is not loaded on graph otherwise
        self.discriminator.predict(discriminatorDatagen.__next__()[0])
        ###

        num_samples = training_data_df.shape[0]
        steps_per_epoch = num_samples//batch_size
        # Train discriminator and generator successively
        for i in range(training_epochs):
            print('[-] Adversarial model training epoch [',i+1,'/',training_epochs,']',sep='')

            # ?????
            from PIL import ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            # ?????

            # Train discriminator
            self.discriminator.fit_generator(discriminatorDatagen, steps_per_epoch=steps_per_epoch,
                epochs=i+1, initial_epoch=i, callbacks=callback, verbose=1)
            # Train adversarial model
            self.generatorTrainModel.fit_generator(generatorDatagen, steps_per_epoch=steps_per_epoch,
                epochs=i+1, initial_epoch=i, verbose=1)

            # Save model every 5 epochs
            if (i % 5 == 0):
                postfix = '_epoch_' + str(i+1)
                print('[*] Saving model at epoch',i+1)
                self.save_model(postfix)


    def save_model(self, postfix=''):
        # Save inference model to file
        if self.generator is not None:
            if (not os.path.isdir('trained_models/')):
                os.makedirs('trained_models/')
            self.generator.save('trained_models/' + self.model_name + postfix + '.h5')
            self.depth_inference_model.save('trained_models/' + self.model_name + '_depth' + postfix + '.h5')
        else:
            print('[!] Model not defined. Abandoning saving.')
            exit()


    def initialize_training(self):
        """
        Initializes generator and discriminator training procedures
        Generator training model is saved in self.generatorTrainModel
        """
        # Generator
        self.create_generator()
        # Discriminator
        self.create_discriminator(omega=self.omega)

        # Compile discriminator model
        adam_opt = Adam(lr=10e-4)
        self.discriminator.compile(adam_opt, loss='binary_crossentropy')

        # Create and compile generator training model
        self.discriminator.trainable = False
        prevReconstructions = [self.generator.output[0], self.generator.output[2],
            self.generator.output[4], self.generator.output[6]]
        nextReconstructions = [self.generator.output[1], self.generator.output[3],
            self.generator.output[5], self.generator.output[7]]
        validationOutputPrev = self.discriminator([self.generator.input[0]] + prevReconstructions)
        validationOutputNext = self.discriminator([self.generator.input[0]] + nextReconstructions)
        self.generatorTrainModel = Model(inputs=self.generator.input,
            outputs=(self.generator.output[8:] + [validationOutputPrev, validationOutputNext]))
        #self.generatorTrainModel.summary()
        loss_list = (['mae' for i in range(4)] + # Photometric errors
                     ['binary_crossentropy' for i in range(2)]) # Adversarial errors
        loss_weights = ([0.995/4 for i in range(4)] + 
                        [0.005/2 for i in range(2)])
        self.generatorTrainModel.compile(adam_opt, loss=loss_list, loss_weights=loss_weights)
        print('[*] Created generator training model')


    def create_generator(self):
        """
        Creates the generator model and saves it in self.generator
        """
        depthEncoder = self.create_depth_encoder()
        #depthEncoder.summary()
        print('[*] Created depth encoder')
        depthDecoder = self.create_depth_decoder()
        #depthDecoder.summary()
        print('[*] Created depth decoder')
        poseNet = self.create_pose_net()
        #poseNet.summary()
        print('[*] Created pose estimation network')
        intrinsicsMatrix = np.array([[1000/2.36, 0, 950/2.36],
            [0, 1000/2.36, 540/2.36], [0, 0, 1]]) # Constant during training
        reprojectionModule = self.create_reprojection_module(intrinsicsMatrix)
        #reprojectionModule.summary()
        print('[*] Created reprojection module')

        # Synthesize generator from modules
        target_frame = Input(shape=self.img_shape)
        prev_frame = Input(shape=self.img_shape)
        next_frame = Input(shape=self.img_shape)
        # Latent depth features of frames
        target_encoded = depthEncoder(target_frame)
        prev_encoded = depthEncoder(prev_frame)
        next_encoded = depthEncoder(next_frame)
        # Inverse depth multi-scale estimations
        inverse_depths = depthDecoder(target_encoded)
        # Pose estimations (ignore skip connection outputs)
        pose_estimation = poseNet([target_encoded[0], prev_encoded[0], next_encoded[0]])
        # Source to target reprojections
        reconstructions = reprojectionModule([prev_frame, next_frame] + inverse_depths + [pose_estimation])
        # Apply per-source minimum to MAE at each scale
        scale1_min_error = Lambda(perReprojectionMinimumMAE)([target_frame, reconstructions[0], reconstructions[1]])
        scale2_min_error = Lambda(perReprojectionMinimumMAE)([target_frame, reconstructions[2], reconstructions[3]])
        scale3_min_error = Lambda(perReprojectionMinimumMAE)([target_frame, reconstructions[4], reconstructions[5]])
        scale4_min_error = Lambda(perReprojectionMinimumMAE)([target_frame, reconstructions[6], reconstructions[7]])
        perScaleErrors = [scale1_min_error, scale2_min_error, scale3_min_error, scale4_min_error]
        # Create and save generator model
        self.generator = Model(inputs=[target_frame, prev_frame, next_frame], outputs=(reconstructions + perScaleErrors))
        # Depth inference model
        self.depth_inference_model = Model(inputs=target_frame, outputs=inverse_depths[0])
        #self.generator.summary()
        print('[*] Created generator')


    def create_discriminator(self, omega):
        """
        Creates the discriminator model and saves it in self.discriminator
        """
        target_img = Input(shape=self.img_shape)
        reprojected_img1 = Input(shape=self.img_shape)
        reprojected_img2 = Input(shape=self.img_shape)
        reprojected_img3 = Input(shape=self.img_shape)
        reprojected_img4 = Input(shape=self.img_shape)

        # Create adversarial input
        #adversarial_input1 = Lambda(generateAdversarialInput,
        #    arguments={'omega': omega})([target_img, reprojected_img1])
        #adversarial_input2 = Lambda(generateAdversarialInput,
        #    arguments={'omega': omega})([target_img, reprojected_img2])
        #adversarial_input3 = Lambda(generateAdversarialInput,
        #    arguments={'omega': omega})([target_img, reprojected_img3])
        #adversarial_input4 = Lambda(generateAdversarialInput,
        #    arguments={'omega': omega})([target_img, reprojected_img4])

        input2 = Conv2D(filters=1, kernel_size=1, padding='same', strides=2)(reprojected_img2)
        input2 = LeakyReLU(alpha=0.3)(input2)
        input3 = Conv2D(filters=1, kernel_size=1, padding='same', strides=4)(reprojected_img3)
        input3 = LeakyReLU(alpha=0.3)(input3)
        input4 = Conv2D(filters=1, kernel_size=1, padding='same', strides=8)(reprojected_img4)
        input4 = LeakyReLU(alpha=0.3)(input4)

        # Discriminator network
        discr1 = Conv2D(filters=512, kernel_size=5, strides=2, padding='same')(reprojected_img1)
        discr1_act = LeakyReLU(alpha=0.3)(discr1)
        scale2 = Concatenate()([discr1_act, input2])
        discr2 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(scale2)
        discr2_act = LeakyReLU(alpha=0.3)(discr2)
        scale3 = Concatenate()([discr2_act, input3])
        discr3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(scale3)
        discr3_act = LeakyReLU(alpha=0.3)(discr3)
        scale4 = Concatenate()([discr3_act, input4])
        discr3_flat = Flatten()(scale4)
        validation = Dense(units=1, activation='sigmoid')(discr3_flat)

        # Create discriminator
        self.discriminator = Model(inputs=[target_img, reprojected_img1, reprojected_img2,
            reprojected_img3, reprojected_img4], outputs=validation)
        #self.discriminator.summary()
        print('[*] Created discriminator')


    def create_depth_encoder(self):
        """
        Creates depth encoder network model
            Outputs:
                depthEncoder: Encoder network model with added skip connection
                outputs from outmost to deepest features
        """
        depthEncoder = ResNet18(input_shape=self.img_shape, weights='imagenet', include_top=False)
        # Chosen skip connections (not tested)
        #depthEncoder.summary()
        skip_connection1 = depthEncoder.get_layer('relu0')
        skip_connection2 = depthEncoder.get_layer('stage2_unit1_relu1')
        skip_connection3 = depthEncoder.get_layer('stage3_unit1_relu1')
        skip_connection4 = depthEncoder.get_layer('stage4_unit1_relu1')

        # Ignore batch dimension of output shapes
        self.latent_depth_features_shape = depthEncoder.output_shape[1:]
        self.skip_connection_shapes = [skip_connection1.output_shape[1:], 
            skip_connection2.output_shape[1:], skip_connection3.output_shape[1:],
            skip_connection4.output_shape[1:]]
        
        return Model(inputs=depthEncoder.input, outputs=[depthEncoder.output,
            skip_connection1.output, skip_connection2.output, skip_connection3.output,
            skip_connection4.output])


    def create_depth_decoder(self):
        """
        Creates depth decoder network model
            Outputs:
                depthDecoder: Model implementing the depth decoder network with 
                disparity outputs from higher to lower resolutions
        """
        latent_depth_features = Input(shape=self.latent_depth_features_shape)
        skip_connection1 = Input(shape=self.skip_connection_shapes[0])
        skip_connection2 = Input(shape=self.skip_connection_shapes[1])
        skip_connection3 = Input(shape=self.skip_connection_shapes[2])
        skip_connection4 = Input(shape=self.skip_connection_shapes[3])

        # Network tower
        upconv5 = Conv2D(filters=256, kernel_size=3, padding='same', activation='elu')(latent_depth_features)
        upsampledConv5 = UpSampling2D(size=(2,2), interpolation='nearest')(upconv5)
        concatInputs = Concatenate()([upsampledConv5, skip_connection4])
        iconv5 = Conv2D(filters=256, kernel_size=3, padding='same', activation='elu')(concatInputs)
        
        upconv4 = Conv2D(filters=128, kernel_size=3, padding='same', activation='elu')(iconv5)
        upsampledConv4 = UpSampling2D(size=(2,2), interpolation='nearest')(upconv4)
        concatInputs = Concatenate()([upsampledConv4, skip_connection3])
        iconv4 = Conv2D(filters=128, kernel_size=3, padding='same', activation='elu')(concatInputs)
        disp4 = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid', name='disp4')(iconv4)
        
        upconv3 = Conv2D(filters=64, kernel_size=3, padding='same', activation='elu')(iconv4)
        upsampledConv3 = UpSampling2D(size=(2,2), interpolation='nearest')(upconv3)
        concatInputs = Concatenate()([upsampledConv3, skip_connection2])
        iconv3 = Conv2D(filters=64, kernel_size=3, padding='same', activation='elu')(concatInputs)
        disp3 = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid', name='disp3')(iconv3)

        upconv2 = Conv2D(filters=32, kernel_size=3, padding='same', activation='elu')(iconv3)
        upsampledConv2 = UpSampling2D(size=(2,2), interpolation='nearest')(upconv2)
        concatInputs = Concatenate()([upsampledConv2, skip_connection1])
        iconv2 = Conv2D(filters=32, kernel_size=3, padding='same', activation='elu')(concatInputs)
        disp2 = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid', name='disp2')(iconv2)

        upconv1 = Conv2D(filters=16, kernel_size=3, padding='same', activation='elu')(iconv2)
        upsampledConv1 = UpSampling2D(size=(2,2), interpolation='nearest')(upconv1)
        iconv1 = Conv2D(filters=16, kernel_size=3, padding='same', activation='elu')(upsampledConv1)
        disp1 = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid', name='disp1')(iconv1)

        return Model(inputs=[latent_depth_features, skip_connection1, skip_connection2,
            skip_connection3, skip_connection4], outputs=[disp1, disp2, disp3, disp4])


    def create_pose_net(self):
        """
        Creates pose estimation network model
                poseNet: Pose estimation network model with output size [batchSize, 12]
                (Output is [tp rp tn rn]^T)
        """
        latent_depth_features_target = Input(shape=self.latent_depth_features_shape)
        latent_depth_features_prev = Input(shape=self.latent_depth_features_shape)
        latent_depth_features_next = Input(shape=self.latent_depth_features_shape)

        # Estimate target to source poses from the latent features
        pconv01 = Conv2D(filters=256, kernel_size=1, padding='same',
            activation='relu')(latent_depth_features_target)
        pconv02 = Conv2D(filters=256, kernel_size=1, padding='same',
            activation='relu')(latent_depth_features_prev)
        pconv03 = Conv2D(filters=256, kernel_size=1, padding='same',
            activation='relu')(latent_depth_features_next)
        concatInputs = Concatenate()([pconv01, pconv02, pconv03])

        pconv1 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu')(concatInputs)
        pconv2 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu')(pconv1)
        pconv3 = Conv2D(filters=12, kernel_size=1, padding='same', activation='linear')(pconv2)
        poses = GlobalAveragePooling2D()(pconv3)

        return Model(inputs=[latent_depth_features_target, latent_depth_features_prev,
            latent_depth_features_next], outputs=[poses])
    

    def create_reprojection_module(self, intrinsicsMatrix):
        """
        Creates differentiable source to target image reprojection module
            Inputs:
                intrinsicsMatrix: The camera intrinsics matrix
            Outputs:
                reprojectionModule: The image reprojection module which outputs frame
                (t-1) and (t+1) reconstructions from highest to lowest quality
        """
        source_img_prev = Input(shape=self.img_shape)
        source_img_next = Input(shape=self.img_shape)
        disp1 = Input(shape=self.img_shape)
        disp2 = Input(shape=tuple(map(lambda x: x//2, self.img_shape)))
        disp3 = Input(shape=tuple(map(lambda x: x//4, self.img_shape)))
        disp4 = Input(shape=tuple(map(lambda x: x//8, self.img_shape)))
        # Split into prev and next frame pose vectors
        pose = Input(shape=(12,))
        pose_prev = Lambda(lambda x: x[:,:6])(pose)
        pose_next = Lambda(lambda x: x[:,6:])(pose)

        # Upsample and convert to depth maps
        disp2Up = UpSampling2D(size=(2,2), interpolation='nearest')(disp2)
        disp3Up = UpSampling2D(size=(4,4), interpolation='nearest')(disp3)
        disp4Up = UpSampling2D(size=(8,8), interpolation='nearest')(disp4)
        depth1 = Lambda(inverseDepthNormalization)(disp1)
        depth2 = Lambda(inverseDepthNormalization)(disp2Up)
        depth3 = Lambda(inverseDepthNormalization)(disp3Up)
        depth4 = Lambda(inverseDepthNormalization)(disp4Up)

        target_reconstructions = []
        for depth in [depth1, depth2, depth3, depth4]:
            prev_to_target = ProjectionLayer(intrinsicsMatrix)([source_img_prev, depth, pose_prev])
            next_to_target = ProjectionLayer(intrinsicsMatrix)([source_img_next, depth, pose_next])
            target_reconstructions += [prev_to_target, next_to_target]

        return Model(inputs=[source_img_prev, source_img_next, disp1, disp2, disp3, disp4, pose],
            outputs=target_reconstructions)

    
    def discriminator_train_datagen(self, training_data_df, height, width, batch_size):
        """
        Creates training data generator for the model discriminator network
            Inputs:
                training_data_df: The dataframe containing the paths to frame triplets
                height: Input image height
                width: Input image width
                batch_size: Generated batch sizes
            Outputs:
                ([inputs], [outputs]) tuple for discriminator training
        """
        # Image preprocessor
        datagen = pre.ImageDataGenerator(
                rescale = 1./255,
                channel_shift_range=0.1,
                fill_mode='nearest')

        # Frame generators
        seed = int(np.random.rand(1,1)*2000) # Using same seed to ensure temporal continuity
        curr_generator = datagen.flow_from_dataframe(training_data_df, directory=None,
            x_col='curr_frame', target_size=(height, width), batch_size=batch_size,
            interpolation='bilinear', class_mode=None, seed=seed)
        prev_generator = datagen.flow_from_dataframe(training_data_df, directory=None,
            x_col='prev_frame', target_size=(height,width), batch_size=batch_size,
            interpolation='bilinear', class_mode=None, seed=seed)
        next_generator = datagen.flow_from_dataframe(training_data_df, directory=None,
            x_col='next_frame', target_size=(height,width), batch_size=batch_size,
            interpolation='bilinear', class_mode=None, seed=seed)
    
        while True:
            curr_frame = curr_generator.__next__()
            prev_frame = prev_generator.__next__()
            next_frame = next_generator.__next__()

            # Use first half of the batch samples as invalid generator outputs
            half_batch = np.ceil(curr_frame.shape[0]/2.).astype(int)
            generator_samples = self.generator.predict([curr_frame[:half_batch],
                prev_frame[:half_batch], next_frame[:half_batch]])
            # Use half of the generator reprojections from prev_frame and half from next_frame
            quarter_batch = np.ceil(half_batch/2.).astype(int)
            generator_split1 = np.concatenate((generator_samples[0][:quarter_batch],
                generator_samples[1][quarter_batch:]), axis=0)
            generator_split2 = np.concatenate((generator_samples[2][:quarter_batch],
                generator_samples[3][quarter_batch:]), axis=0)
            generator_split3 = np.concatenate((generator_samples[4][:quarter_batch],
                generator_samples[5][quarter_batch:]), axis=0)
            generator_split4 = np.concatenate((generator_samples[6][:quarter_batch],
                generator_samples[7][quarter_batch:]), axis=0)
            # Create input batches
            reprojected_imgs1 = np.concatenate((generator_split1, curr_frame[half_batch:]), axis=0)
            reprojected_imgs2 = np.concatenate((generator_split2, curr_frame[half_batch:]), axis=0)
            reprojected_imgs3 = np.concatenate((generator_split3, curr_frame[half_batch:]), axis=0)
            reprojected_imgs4 = np.concatenate((generator_split4, curr_frame[half_batch:]), axis=0)
            # Rest of the batch are valid samples 
            validation_labels = np.concatenate((np.zeros((half_batch,1)),
                np.ones((curr_frame.shape[0]-half_batch,1))), axis=0)

            # Returns (inputs, outputs) tuple
            yield ([curr_frame, reprojected_imgs1, reprojected_imgs2,
                reprojected_imgs3, reprojected_imgs4], validation_labels)


    def generator_train_datagen(self, training_data_df, height, width, batch_size):
        """
        Creates tranining data generator for the model generator network
            Inputs:
                training_data_df: The dataframe containing the paths to frame triplets
                height: Input image height
                width: Input image width
                batch_size: Generated batch sizes
            Outputs:
                ([inputs], [outputs]) tuple for generator training
        """
        # Image preprocessor
        datagen = pre.ImageDataGenerator(
                rescale = 1./255,
                channel_shift_range=0.1,
                fill_mode='nearest')

        # Frame generators
        seed = int(np.random.rand(1,1)*1000) # Using same seed to ensure temporal continuity
        curr_generator = datagen.flow_from_dataframe(training_data_df, directory=None,
            x_col='curr_frame', target_size=(height,width), batch_size=batch_size,
            interpolation='bilinear', class_mode=None, seed=seed)
        prev_generator = datagen.flow_from_dataframe(training_data_df, directory=None,
            x_col='prev_frame', target_size=(height,width), batch_size=batch_size,
            interpolation='bilinear', class_mode=None, seed=seed)
        next_generator = datagen.flow_from_dataframe(training_data_df, directory=None,
            x_col='next_frame', target_size=(height,width), batch_size=batch_size,
            interpolation='bilinear', class_mode=None, seed=seed)
    
        while True:
            curr_frame = curr_generator.__next__()
            prev_frame = prev_generator.__next__()
            next_frame = next_generator.__next__()

            # Returns (inputs,outputs) tuple
            batch_size_curr = curr_frame.shape[0]
            zeros_maps = np.zeros((4, batch_size_curr, height, width, 1)) # Zero min per-scale error
            true_labels = np.ones((2, batch_size_curr, 1))

            yield ([curr_frame, prev_frame, next_frame], (zeros_maps.tolist() + true_labels.tolist()))



if __name__ == '__main__':
    model = MonocularAdversarialModel()

    if model.args.train == True:
        model.initialize_training()
        model.train(model.args.train_dataframe, model.args.batch_size, model.args.training_epochs)
        model.save_model()
    else:
        print('[!] Unknown operation argument, use -h flag for help.')

