import argparse
import numpy as np
import pandas as pd
import os
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D, GlobalAveragePooling2D, \
    LeakyReLU, Dense, Flatten, Input, Concatenate, Lambda, BatchNormalization
from keras.optimizers import Adam
from keras import callbacks
import keras.preprocessing.image as pre
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
        parser.add_argument('--save_per', type=int, help='Epochs between saving model during training.', default=5)
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
        adversarialDatagen = self.adversarial_datagen(training_data_df, self.img_shape[0], self.img_shape[1],
            batch_size)
        discriminatorValidDatagen = self.discriminator_valid_datagen(training_data_df, self.img_shape[0], self.img_shape[1],
            batch_size)
        discriminatorInvalidDatagen = self.discriminator_invalid_datagen(training_data_df, self.img_shape[0], self.img_shape[1],
            batch_size)

        # Keras bug - model is not loaded on graph otherwise
        self.generator.predict(adversarialDatagen.__next__()[0])
        self.discriminator.predict(discriminatorValidDatagen.__next__()[0])
        self.discriminator.predict(discriminatorInvalidDatagen.__next__()[0])
        ###

        num_samples = training_data_df.shape[0]
        steps_per_epoch = num_samples//batch_size
        # Train discriminator and generator successively
        for i in range(training_epochs):
            print('[-] Adversarial model training epoch [',i+1,'/',training_epochs,']',sep='')
            print('[-] Discriminator Valid')
            #self.discriminator.fit_generator(discriminatorValidDatagen, steps_per_epoch=steps_per_epoch,
            #    epochs=i+1, initial_epoch=i, verbose=1)
            print('[-] Discriminator Invalid')
            #self.discriminator.fit_generator(discriminatorInvalidDatagen, steps_per_epoch=steps_per_epoch,
            #    epochs=i+1, initial_epoch=i, verbose=1)
            print('[-] Adversarial model')
            self.adversarial_model.fit_generator(adversarialDatagen, steps_per_epoch=steps_per_epoch,
                epochs=i+1, initial_epoch=i, verbose=1)

            # Save model every x epochs
            if (i % self.args.save_per == 0):
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
        Adversarial training model is saved in self.adversarialModel
        """
        # Create generator and discriminator
        self.create_generator()
        print('[*] Created generator model')
        self.create_discriminator(self.omega)
        print('[*] Created discriminator model')
        # Synthesize adversarial model
        prevReprojections = [self.generator.output[0], self.generator.output[2],
                             self.generator.output[4], self.generator.output[6]]
        nextReprojections = [self.generator.output[1], self.generator.output[3],
                             self.generator.output[5], self.generator.output[7]]
        inverseDepths = [self.generator.output[8], self.generator.output[9],
                             self.generator.output[10], self.generator.output[11]]
        # Concatenate reprojections per-scale for computing loss
        perScaleReprojections = \
            [Concatenate(name='scale1_reprojections')([prevReprojections[0], nextReprojections[0]]),
             Concatenate(name='scale2_reprojections')([prevReprojections[1], nextReprojections[1]]),
             Concatenate(name='scale3_reprojections')([prevReprojections[2], nextReprojections[2]]),
             Concatenate(name='scale4_reprojections')([prevReprojections[3], nextReprojections[3]])]
        # Adversarial error
        validationPrev = self.discriminator([self.generator.input[0]] + prevReprojections)
        validationNext = self.discriminator([self.generator.input[0]] + nextReprojections)
        # Adversarial model
        self.adversarial_model = Model(inputs=self.generator.input,
            outputs=(perScaleReprojections))
            # + [validationPrev, validationNext]))

        # Compile models
        adam_opt = Adam(lr=10e-4)
        self.discriminator.compile(adam_opt, loss='binary_crossentropy')

        self.discriminator.trainable = False
        loss_list = ([perScaleMinMAE for _ in range(4)])
                     #[None for _ in range(2)])
        loss_weights = ([1 for _ in range(4)])
                        #[0.005/2 for _ in range(2)])
        self.adversarial_model.compile(adam_opt, loss=loss_list, loss_weights=loss_weights)
        print('[*] Created adversarial model')


    def create_generator(self):
        """
        Creates the generator and depth predicting models
        """
        # Generator modules
        depth_encoder, depth_encoder_skip = self.create_depth_encoder()
        depth_decoder = self.create_depth_decoder(depth_encoder_skip.output_shape)
        pose_net = self.create_pose_net(depth_encoder.output_shape)
        reprojection_module = self.create_reprojection_module()

        # Synthesize generator model
        curr_frame = Input(shape=self.img_shape)
        prev_frame = Input(shape=self.img_shape)
        next_frame = Input(shape=self.img_shape)
        # Depth features 
        depth_encoder_curr = depth_encoder_skip(curr_frame)
        depth_features_prev = depth_encoder(prev_frame)
        depth_features_next = depth_encoder(next_frame)
        # Inverse depth and pose
        inverseDepths = depth_decoder(depth_encoder_curr)
        poses = pose_net([depth_encoder_curr[0], depth_features_prev, depth_features_next])
        # Reprojections
        reprojections = reprojection_module([prev_frame, next_frame, poses] + inverseDepths)
        # Models
        self.generator = Model(inputs=[curr_frame, prev_frame, next_frame], outputs=(reprojections + inverseDepths))
        self.depth_inference_model = Model(inputs=[curr_frame, prev_frame, next_frame], outputs=inverseDepths)


    def create_depth_encoder(self):
        """
        Creates and returns the depth encoder model
            Outputs:
                depth_encoder: Depth encoder model
                depth_encoder_skip: Depth encoder model also outputting skip connections
        """
        # Layer-generating function
        def enc_layer(prev_layer, filters, stride, kernel=3):
            enc = Conv2D(filters=filters, kernel_size=kernel, strides=stride, padding='same', activation='relu')(prev_layer) 
            return enc

        # Input
        input_frame = Input(shape=self.img_shape)
        # Layers 
        enc1 =  enc_layer(input_frame, 32,  stride=2, kernel=7)
        enc1b = enc_layer(enc1,  32,  stride=1, kernel=7)
        enc2 =  enc_layer(enc1b, 64,  stride=2, kernel=5)
        enc2b = enc_layer(enc2,  64,  stride=1, kernel=5)
        enc3 =  enc_layer(enc2b, 128, stride=2)
        enc3b = enc_layer(enc3,  128, stride=1)
        enc4 =  enc_layer(enc3b, 256, stride=2)
        enc4b = enc_layer(enc4,  256, stride=1)
        enc5 =  enc_layer(enc4b, 512, stride=2)
        enc5b = enc_layer(enc5,  512, stride=1)
        enc6 =  enc_layer(enc5b, 512, stride=2)
        enc6b = enc_layer(enc6,  512, stride=1)
        enc7 =  enc_layer(enc6b, 512, stride=2)
        enc7b = enc_layer(enc7,  512, stride=1)
        # Model
        depth_encoder = Model(inputs=[input_frame], outputs=[enc7b])
        depth_encoder_skip = Model(inputs=[input_frame],
            outputs=[enc7b, enc6b, enc5b, enc4b, enc3b, enc2b, enc1b]) 
        return depth_encoder, depth_encoder_skip


    def create_depth_decoder(self, depth_encoder_output_shapes):
        """
        Creates and returns the depth decoder model
            Inputs:
                depth_encoder_outputs_shapes: Depth encoder output and skip
                connection shapes
            Outputs:
                depth_decoder: Depth decoder model
        """
        # Layer-generating functions
        def dec_layer(prev_layer, skip_layer, filters, upsample):
            dec = prev_layer
            if (upsample):
                dec = UpSampling2D(size=(2,2))(dec)
            if (skip_layer != None):
                dec = Concatenate()([skip_layer, dec])
            dec = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation='sigmoid')(dec)
            return dec

        def generate_disparity(prev_layer, prev_disp=None):
            if prev_disp is not None:
                prev_disp_up = UpSampling2D(size=(2,2), interpolation='nearest')(prev_disp)
                prev_layer = Concatenate()([prev_layer, prev_disp_up])
            disp = Conv2D(filters=1, kernel_size=3, padding='same', activation='relu')(prev_layer)
            return disp

        # Inputs
        depth_features = Input(shape=depth_encoder_output_shapes[0][1:])
        skip6 = Input(shape=depth_encoder_output_shapes[1][1:])
        skip5 = Input(shape=depth_encoder_output_shapes[2][1:])
        skip4 = Input(shape=depth_encoder_output_shapes[3][1:])
        skip3 = Input(shape=depth_encoder_output_shapes[4][1:])
        skip2 = Input(shape=depth_encoder_output_shapes[5][1:])
        skip1 = Input(shape=depth_encoder_output_shapes[6][1:])
        # Layers
        dec7  = dec_layer(depth_features, None, 512, upsample=True)
        dec7b = dec_layer(dec7, skip6, 512, upsample=False)
        dec6  = dec_layer(dec7b, None, 512, upsample=True)
        dec6b = dec_layer(dec6, skip5, 512, upsample=False)
        dec5  = dec_layer(dec6b, None, 256, upsample=True)
        dec5b = dec_layer(dec5, skip4, 256, upsample=False)
        dec4  = dec_layer(dec5b, None, 128, upsample=True)
        dec4b = dec_layer(dec4, skip3, 128, upsample=False)
        dec3  = dec_layer(dec4b, None, 64,  upsample=True)
        dec3b = dec_layer(dec3, skip2, 64,  upsample=False)
        dec2  = dec_layer(dec3b, None, 32,  upsample=True)
        dec2b = dec_layer(dec2, skip1, 32,  upsample=False)
        dec1  = dec_layer(dec2b, None, 16,  upsample=True)
        dec1b = dec_layer(dec1,  None, 16,  upsample=False)
        # Inverse depth outputs
        inverseDepth4 = generate_disparity(dec4b)
        inverseDepth3 = generate_disparity(dec3b, inverseDepth4)
        inverseDepth2 = generate_disparity(dec2b, inverseDepth3)
        inverseDepth1 = generate_disparity(dec1b, inverseDepth2)
        # Model
        depth_decoder = Model(inputs=[depth_features, skip6, skip5, skip4, skip3, skip2, skip1],
            outputs=[inverseDepth1, inverseDepth2, inverseDepth3, inverseDepth4])
        return depth_decoder


    def create_pose_net(self, depth_encoder_latent_shape):
        """
        Creates and returns the pose estimation network
            Inputs:
                depth_encoder_latent_shape: Depth encoder latent features shape
            Outputs:
                pose_net: Pose estimation network model
        """
        # Input depth features
        depth_features_curr = Input(shape=depth_encoder_latent_shape[1:])
        depth_features_prev = Input(shape=depth_encoder_latent_shape[1:])
        depth_features_next = Input(shape=depth_encoder_latent_shape[1:])
        # Estimate target to source poses from latent depth features
        pconv0curr = Conv2D(filters=256, kernel_size=1, padding='same', activation='relu')(depth_features_curr)
        pconv0prev = Conv2D(filters=256, kernel_size=1, padding='same', activation='relu')(depth_features_prev)
        pconv0next = Conv2D(filters=256, kernel_size=1, padding='same', activation='relu')(depth_features_next)

        poseFeatures = Concatenate()([pconv0curr, pconv0prev, pconv0next])
        pconv1 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu')(poseFeatures)
        pconv2 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu')(pconv1)
        pconv3 = Conv2D(filters=12, kernel_size=1, padding='same', activation='linear')(pconv2)
        poses = GlobalAveragePooling2D()(pconv3)
        pose_net = Model(inputs=[depth_features_curr, depth_features_prev, depth_features_next], outputs=poses)
        return pose_net


    def create_reprojection_module(self):
        """
        Creates and returns the reprojection module model
            Outputs:
                reprojection_module: Reprojection module model
        """
        prev_frame = Input(shape=self.img_shape)
        next_frame = Input(shape=self.img_shape)
        poses = Input(shape=(12,))
        inverseDepth1 = Input(shape=(self.img_shape[0], self.img_shape[1], 1))
        inverseDepth2 = Input(shape=(self.img_shape[0]//2, self.img_shape[1]//2, 1))
        inverseDepth3 = Input(shape=(self.img_shape[0]//4, self.img_shape[1]//4, 1))
        inverseDepth4 = Input(shape=(self.img_shape[0]//8, self.img_shape[1]//8, 1))

        posePrev = Lambda(lambda x: x[:,:6])(poses)
        poseNext = Lambda(lambda x: x[:,6:])(poses)
        intrinsicsMatrix = np.array([[1000/2.36, 0, 950/2.36],
            [0, 1000/2.36, 540/2.36], [0, 0, 1]])
        #intrinsicsMatrix = np.array([[1, 0, 0.5],
        #    [0, 1, 0.5], [0, 0, 1]])

        # Upsample and normalize inverse depth maps
        inverseDepth2Up = UpSampling2D(size=(2,2), interpolation='nearest')(inverseDepth2)
        inverseDepth3Up = UpSampling2D(size=(4,4), interpolation='nearest')(inverseDepth3)
        inverseDepth4Up = UpSampling2D(size=(8,8), interpolation='nearest')(inverseDepth4)
        depthMap1 = Lambda(inverseDepthNormalization)(inverseDepth1)
        depthMap2 = Lambda(inverseDepthNormalization)(inverseDepth2Up)
        depthMap3 = Lambda(inverseDepthNormalization)(inverseDepth3Up)
        depthMap4 = Lambda(inverseDepthNormalization)(inverseDepth4Up)

        # Create reprojections for each depth map scale from highest to lowest
        reprojections = []
        for depthMap in [depthMap1, depthMap2, depthMap3, depthMap4]:
            prevToTarget = ProjectionLayer(intrinsicsMatrix)([prev_frame, depthMap, posePrev])
            nextToTarget = ProjectionLayer(intrinsicsMatrix)([next_frame, depthMap, poseNext])
            reprojections += [prevToTarget, nextToTarget]

        reprojection_module = Model(inputs=[prev_frame, next_frame, poses,
            inverseDepth1, inverseDepth2, inverseDepth3, inverseDepth4],
            outputs=reprojections)
        return reprojection_module


    def create_discriminator(self, omega):
        """
        Creates the discriminator model and saves it in self.discriminator
        """
        # Layer-generating function
        def discr_layer(prev_layer, filters, kernel=3, skip=None, batch_norm=True):
            if skip is not None:
                prev_layer = Concatenate()([prev_layer, skip])
            discr = Conv2D(filters=filters, kernel_size=kernel, strides=2, padding='same')(prev_layer) 
            if batch_norm:
                discr = BatchNormalization()(discr)
            discr = LeakyReLU(alpha=0.2)(discr)
            return discr
            
        #Inputs
        target_img = Input(shape=self.img_shape)
        reprojected_img1 = Input(shape=self.img_shape)
        reprojected_img2 = Input(shape=self.img_shape)
        reprojected_img3 = Input(shape=self.img_shape)
        reprojected_img4 = Input(shape=self.img_shape)
        # Downsample input images
        skip2 = Conv2D(filters=1, kernel_size=1, strides=2, padding='same')(reprojected_img2)
        skip2 = LeakyReLU(alpha=0.2)(skip2)
        skip3 = Conv2D(filters=1, kernel_size=1, strides=4, padding='same')(reprojected_img3)
        skip3 = LeakyReLU(alpha=0.2)(skip3)
        skip4 = Conv2D(filters=1, kernel_size=1, strides=8, padding='same')(reprojected_img4)
        skip4 = LeakyReLU(alpha=0.2)(skip4)
        #adversarial_input = Lambda(generateAdversarialInput,
        #    arguments={'omega': omega})([target_img, reprojected_img])
        # Discriminator network
        discr1  = discr_layer(reprojected_img1, 32, kernel=5, batch_norm=False)
        discr2  = discr_layer(discr1,  64, skip=skip2)
        discr3  = discr_layer(discr2, 128, skip=skip3)
        discr4  = discr_layer(discr3, 256, skip=skip4)
        discr5  = discr_layer(discr4, 512)
        discr6  = discr_layer(discr5, 512)

        discr6_flat = Flatten()(discr6)
        validation = Dense(units=1, activation='sigmoid')(discr6_flat)

        # Create discriminator
        self.discriminator = Model(inputs=[target_img, reprojected_img1, reprojected_img2,
            reprojected_img3, reprojected_img4], outputs=validation)
        #self.discriminator.summary()


    def discriminator_invalid_datagen(self, training_data_df, height, width, batch_size):
        """
        Creates invalid training data generator for the discriminator model 
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
            horizontal_flip=True,
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

            generator_samples = self.generator.predict([curr_frame,
                prev_frame, next_frame])
            # Use half of the generator reprojections from prev_frame and half from next_frame
            half_batch = curr_frame.shape[0]//2
            generator_split1 = np.concatenate((generator_samples[0][:half_batch],
                generator_samples[1][half_batch:]), axis=0)
            generator_split2 = np.concatenate((generator_samples[2][:half_batch],
                generator_samples[3][half_batch:]), axis=0)
            generator_split3 = np.concatenate((generator_samples[4][:half_batch],
                generator_samples[5][half_batch:]), axis=0)
            generator_split4 = np.concatenate((generator_samples[6][:half_batch],
                generator_samples[7][half_batch:]), axis=0)
            validation_labels = np.zeros((generator_samples[0].shape[0],1))

            # Returns (inputs, outputs) tuple
            yield ([curr_frame, generator_split1, generator_split2, generator_split3,
                generator_split4], validation_labels)


    def discriminator_valid_datagen(self, training_data_df, height, width, batch_size):
        """
        Creates valid training data generator for the discriminator model
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
            horizontal_flip=True,
            fill_mode='nearest')

        # Frame generator
        seed = int(np.random.rand(1,1)*3000) # Using same seed to ensure temporal continuity
        curr_generator = datagen.flow_from_dataframe(training_data_df, directory=None,
            x_col='curr_frame', target_size=(height, width), batch_size=batch_size,
            interpolation='bilinear', class_mode=None, seed=seed)
    
        while True:
            curr_frame = curr_generator.__next__()
            curr_frame_list = [curr_frame for _ in range(5)]
            validation_labels = np.ones((curr_frame.shape[0],1))

            # Returns (inputs, outputs) tuple
            yield (curr_frame_list, validation_labels)


    def adversarial_datagen(self, training_data_df, height, width, batch_size):
        """
        Creates tranining data generator for the adversarial model
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
            horizontal_flip=True,
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
            curr_frame_list = [curr_frame for _ in range(4)]
            true_labels = [np.ones((batch_size_curr, 1)) for _ in range(2)]

            yield ([curr_frame, prev_frame, next_frame], curr_frame_list)



if __name__ == '__main__':
    model = MonocularAdversarialModel()

    if model.args.train == True:
        model.initialize_training()
        model.train(model.args.train_dataframe, model.args.batch_size, model.args.training_epochs)
        print('[*] Saving model')
        model.save_model()
    else:
        print('[!] Unknown operation argument, use -h flag for help.')

