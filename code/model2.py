import argparse
import numpy as np
import pandas as pd
import os
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D, GlobalAveragePooling2D, \
    LeakyReLU, Dense, Flatten, Input, Concatenate, Lambda, BatchNormalization
from classification_models.resnet import ResNet18
from keras.optimizers import Adam
from keras import callbacks
import keras.preprocessing.image as pre
import extras

class MonocularAdversarialModel:

    def __init__(self):
        """"
        Read command-line arguments and initialize general model parameters
        """
        self.args = self.parse_args()
        # Model parameters
        self.img_shape = (self.args.img_height, self.args.img_width, 3)
        self.model_name = self.args.model_name


    def parse_args(self):
        """
        Parse command-line input arguments
            Outputs:
                args: The arguments object
        """
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Model training script.')
        parser.add_argument('--train', action='store_true', help='Train model.', default=False)
        parser.add_argument('--train_dataframe', type=str, help='File containing the training dataFrame.', default=None)
        parser.add_argument('--batch_size', type=int, help='Training batch size.', default=8)
        parser.add_argument('--training_epochs', type=int, help='Model training epochs.', default=10)
        parser.add_argument('--model_name', type=str, help='Name to use for saving model.', default='monovideo')
        parser.add_argument('--img_width', type=int, help='Input image width.', default=224)
        parser.add_argument('--img_height', type=int, help='Input image height.', default=128)
        parser.add_argument('--save_per', type=int, help='Epochs between saving model during training.', default=5)
        args = parser.parse_args()
        return args


    # Learning rate halving callback function
    def __learning_rate_halving(self, start_epoch, period):
        """
        Create callback function to halve learning rate during training
            Inputs:
                start_epoch: First epoch to halve learning rate at
                period: Learning rate halving epoch period
            Outputs:
                halve_lr: Learning rate halving callback function
        """
        def halve_lr(epoch, curr_lr):
            # Epochs are zero-indexed
            if (epoch < start_epoch-1):
                return curr_lr
            else:
                if ((epoch-(start_epoch-1)) % period == 0):
                    print('[*] Halving learning rate (=',curr_lr,' -> ',(curr_lr / 2.0),')',sep='')
                    return curr_lr / 2.0
                else:
                    return curr_lr
        return halve_lr


    def train(self, train_df_file, batch_size, training_epochs):
        """
        Train model
            Inputs:
                train_df_file: Training data dataFrame file (.h5, 'df' key)
                batch_size: Training batch size
                training_epochs: Number of training epochs
        """
        # Learning rate halving callback
        learning_rate_callback = self.__learning_rate_halving(10, 5)
        callbacks_list = [callbacks.LearningRateScheduler(schedule=learning_rate_callback, verbose=0)]
        
        training_data_df = pd.read_hdf(train_df_file, 'df')
        num_samples = training_data_df.shape[0]
        steps_per_epoch = 1 #num_samples//batch_size
        # Training data generators
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
        #####################################################

        # Train discriminator and adversarial model successively
        for e in range(1, training_epochs+1):
            print('[-] Adversarial model training epoch [',e,'/',training_epochs,']',sep='')
            print('[-] Discriminator Valid')
            #self.discriminator.fit_generator(discriminatorValidDatagen, steps_per_epoch=steps_per_epoch,
            #    epochs=e, initial_epoch=e-1, callbacks=callbacks_list, verbose=1)
            print('[-] Discriminator Invalid')
            #self.discriminator.fit_generator(discriminatorInvalidDatagen, steps_per_epoch=steps_per_epoch,
            #    epochs=e, initial_epoch=e-1, verbose=1)
            print('[-] Adversarial model')
            self.adversarial_model.fit_generator(adversarialDatagen, steps_per_epoch=steps_per_epoch,
                epochs=e, initial_epoch=e-1, verbose=1)

            # Save model every x epochs
            if (e % self.args.save_per == 0):
                postfix = '_epoch_' + str(e)
                print('[*] Saving model at epoch',e)
                self.save_model(postfix)


    def save_model(self, postfix=''):
        # Save inference model to file
        if self.generator is not None:
            if (not os.path.isdir('trained_models/')):
                os.makedirs('trained_models/')
            self.generator.save('trained_models/' + self.model_name + postfix + '.h5')
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
        self.create_discriminator()
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
        #validationPrev = self.discriminator([self.generator.input[0]] + prevReprojections)
        #validationNext = self.discriminator([self.generator.input[0]] + nextReprojections)
        # Adversarial model
        self.adversarial_model = Model(inputs=self.generator.input, outputs=perScaleReprojections)
        print('[*] Created adversarial model')

        # Compile models
        # Discriminator
        adam_opt = Adam(lr=10e-4)
        self.discriminator.compile(adam_opt, loss='binary_crossentropy')
        # Adversarial
        self.discriminator.trainable = False
        loss_list = ([extras.perScaleMinMAE for _ in range(4)])
        loss_weights = ([1/4 for _ in range(4)])
        self.adversarial_model.compile(adam_opt, loss=loss_list, loss_weights=loss_weights)


    def create_generator(self):
        """
        Creates the generator and depth predicting models
        """
        # Generator modules
        depth_encoder = self.create_depth_encoder()
        depth_decoder = self.create_depth_decoder(depth_encoder.output_shape)
        pose_net = self.create_pose_net()
        reprojection_module = self.create_reprojection_module()

        # Synthesize generator model
        curr_frame = Input(shape=self.img_shape)
        prev_frame = Input(shape=self.img_shape)
        next_frame = Input(shape=self.img_shape)
        # Depth features 
        depth_features = depth_encoder(curr_frame)
        inverseDepths = depth_decoder(depth_features)
        poses = pose_net([curr_frame, prev_frame, next_frame])
        # Reprojections
        reprojections = reprojection_module([prev_frame, next_frame, poses] + inverseDepths)
        # Models
        self.generator = Model(inputs=[curr_frame, prev_frame, next_frame], outputs=(reprojections + inverseDepths))


    def create_depth_encoder(self):
        """
        Creates and returns the depth encoder model
            Outputs:
                depth_encoder: Depth encoder model
        """
        # ResNet18 encoder
        depth_encoder = ResNet18(input_shape=self.img_shape, weights='imagenet', include_top=False)
        skip1 = depth_encoder.get_layer('relu0')
        skip2 = depth_encoder.get_layer('stage2_unit1_relu1')
        skip3 = depth_encoder.get_layer('stage3_unit1_relu1')
        skip4 = depth_encoder.get_layer('stage4_unit1_relu1')
        # Model
        depth_encoder_skip = Model(inputs=depth_encoder.input,
            outputs=[depth_encoder.output, skip4.output, skip3.output, skip2.output, skip1.output]) 
        return depth_encoder_skip


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
            if upsample:
                dec = UpSampling2D(size=(2,2))(dec)
            if skip_layer is not None:
                dec = Concatenate()([skip_layer, dec])
            dec = Conv2D(filters=filters, kernel_size=3, padding='same', activation='elu')(dec)
            return dec

        # Inputs
        depth_features = Input(shape=depth_encoder_output_shapes[0][1:])
        skip4 = Input(shape=depth_encoder_output_shapes[1][1:])
        skip3 = Input(shape=depth_encoder_output_shapes[2][1:])
        skip2 = Input(shape=depth_encoder_output_shapes[3][1:])
        skip1 = Input(shape=depth_encoder_output_shapes[4][1:])
        # Layers
        upconv5 = dec_layer(depth_features, None, 256, False)
        iconv5  = dec_layer(upconv5, skip4, 256, True)
        upconv4 = dec_layer(iconv5,  None,  128, False)
        iconv4  = dec_layer(upconv4, skip3, 128, True)
        upconv3 = dec_layer(iconv4,  None,   64, False)
        iconv3  = dec_layer(upconv3, skip2,  64, True)
        upconv2 = dec_layer(iconv3,  None,   32, False)
        iconv2  = dec_layer(upconv2, skip1,  32, True)
        upconv1 = dec_layer(iconv2,  None,   16, False)
        iconv1  = dec_layer(upconv1, None,   16, True)
        # Inverse depth outputs
        inverseDepth4 = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid')(iconv4)
        inverseDepth3 = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid')(iconv3)
        inverseDepth2 = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid')(iconv2)
        inverseDepth1 = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid')(iconv1)
        # Model
        depth_decoder = Model(inputs=[depth_features, skip4, skip3, skip2, skip1],
            outputs=[inverseDepth1, inverseDepth2, inverseDepth3, inverseDepth4])
        return depth_decoder


    def create_pose_net(self):
        """
        Creates and returns the pose estimation network
            Outputs:
                pose_net: Pose estimation network model
        """
        # Pose encoder
        pose_encoder = ResNet18(input_shape=self.img_shape, weights='imagenet', include_top=False)
        # Pose decoder
        pose_features = Input(shape=pose_encoder.output_shape[1:])
        pconv0 = Conv2D(filters=256, kernel_size=1, padding='same', activation='relu')(pose_features)
        pconv1 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(pconv0)
        pconv2 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(pconv1)
        pconv3 = Conv2D(filters=6,   kernel_size=1, padding='same', activation='linear')(pconv2)
        pose = GlobalAveragePooling2D()(pconv3)
        pose_decoder = Model(inputs=pose_features, outputs=pose)

        # Inputs
        frame_curr = Input(shape=self.img_shape)
        frame_prev = Input(shape=self.img_shape)
        frame_next = Input(shape=self.img_shape)
        adjacent_frames_prev = Concatenate()([frame_prev, frame_curr])
        adjacent_frames_next = Concatenate()([frame_curr, frame_next])
        # Convert concatenated frames to 3-channel volume
        input_pre_prev = Conv2D(filters=3, kernel_size=3, padding='same', activation='relu')(adjacent_frames_prev)
        input_pre_next = Conv2D(filters=3, kernel_size=3, padding='same', activation='relu')(adjacent_frames_next)
        # Estimates poses
        pose_prev = pose_decoder(pose_encoder(input_pre_prev))
        pose_next = pose_decoder(pose_encoder(input_pre_next))
        poses = Concatenate()([pose_prev, pose_next])
        # Model
        pose_net = Model(inputs=[frame_curr, frame_prev, frame_next], outputs=poses)
        #pose_est.summary()
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
        #intrinsicsMatrix = np.array([[0.021, 0, self.img_shape[1]//2],
        #    [0, 0.021, self.img_shape[0]//2], [0, 0, 1]])

        # Upsample and normalize inverse depth maps
        inverseDepth2Up = UpSampling2D(size=(2,2), interpolation='nearest')(inverseDepth2)
        inverseDepth3Up = UpSampling2D(size=(4,4), interpolation='nearest')(inverseDepth3)
        inverseDepth4Up = UpSampling2D(size=(8,8), interpolation='nearest')(inverseDepth4)
        depthMap1 = Lambda(extras.inverseDepthNormalization)(inverseDepth1)
        depthMap2 = Lambda(extras.inverseDepthNormalization)(inverseDepth2Up)
        depthMap3 = Lambda(extras.inverseDepthNormalization)(inverseDepth3Up)
        depthMap4 = Lambda(extras.inverseDepthNormalization)(inverseDepth4Up)

        # Create reprojections for each depth map scale from highest to lowest
        reprojections = []
        for depthMap in [depthMap1, depthMap2, depthMap3, depthMap4]:
            prevToTarget = extras.ProjectionLayer(intrinsicsMatrix)([prev_frame, depthMap, posePrev])
            nextToTarget = extras.ProjectionLayer(intrinsicsMatrix)([next_frame, depthMap, poseNext])
            reprojections += [prevToTarget, nextToTarget]

        reprojection_module = Model(inputs=[prev_frame, next_frame, poses,
            inverseDepth1, inverseDepth2, inverseDepth3, inverseDepth4],
            outputs=reprojections)
        return reprojection_module


    def create_discriminator(self):
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

