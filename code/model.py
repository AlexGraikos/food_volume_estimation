import os
import argparse
import numpy as np
import pandas as pd
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import keras.preprocessing.image as pre
import json
from custom_modules import *
from networks import Networks

class MonocularAdversarialModel:
    def __init__(self):
        """"
        Reads command-line arguments and initialize general model parameters.
        """
        self.args = self.parse_args()
        # Model parameters
        self.img_shape = (self.args.img_height, self.args.img_width, 3)
        self.nets = Networks(self.img_shape)
        self.model_name = self.args.model_name


    def parse_args(self):
        """
        Parses command-line input arguments.
            Outputs:
                args: The arguments object.
        """
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Model training script.')
        parser.add_argument('--train', action='store_true', 
                            help='Train model.',
                            default=False)
        parser.add_argument('--train_dataframe', type=str, 
                            help='File containing the training dataFrame.',
                            default=None)
        parser.add_argument('--batch_size', type=int, 
                            help='Training batch size.',
                            default=8)
        parser.add_argument('--training_epochs', type=int, 
                            help='Model training epochs.',
                            default=10)
        parser.add_argument('--model_name', type=str, 
                            help='Name to use for saving model.',
                            default='monovideo')
        parser.add_argument('--img_width', type=int, 
                            help='Input image width.',
                            default=224)
        parser.add_argument('--img_height', type=int, 
                            help='Input image height.',
                            default=128)
        parser.add_argument('--save_per', type=int, 
                            help='Epochs between model saving.',
                            default=5)
        args = parser.parse_args()
        return args


    def initialize_training(self):
        """
        Initializes discriminator and adversarial model training.
        """
        # Create generator and discriminator models
        self.generator = self.nets.create_generator()
        print('[*] Created generator model')
        self.save_model(self.generator, self.model_name, 'architecture')
        self.discriminator = self.nets.create_discriminator()
        print('[*] Created discriminator model')

        # Synthesize adversarial model
        reprojections = self.generator.outputs[:8]
        inverse_depths = self.generator.outputs[8:12]
        per_scale_reprojections = self.generator.outputs[12:]
        self.adversarial = Model(inputs=self.generator.input,
                                 outputs=(per_scale_reprojections
                                          + inverse_depths),
                                 name='adversarial')
        print('[*] Created adversarial model')

        # Compile models
        adam_opt = Adam(lr=1e-4)
        self.discriminator.compile(adam_opt, loss='binary_crossentropy')

        self.discriminator.trainable = False
        custom_losses = Losses()
        loss_list = ([custom_losses.reprojection_loss(masking=False) 
                      for _ in range(4)] 
                     + [custom_losses.depth_smoothness() for _ in range(4)])
        loss_weights = ([1 for _ in range(4)]
                        + [0.000 for _ in range(4)])
        self.adversarial.compile(adam_opt, loss=loss_list,
                                 loss_weights=loss_weights)


    def train(self, train_df_file, batch_size, training_epochs):
        """
        Trains models.
            Inputs:
                train_df_file: Training data dataFrame file (.h5, 'df' key).
                batch_size: Training batch size.
                training_epochs: Number of training epochs.
        """
        # Learning rate halving callback
        lr_callback = self.__learning_rate_halving(start_epoch=10,
                                                              period=5)
        callbacks_list = [LearningRateScheduler(schedule=lr_callback,
                                                verbose=0)]
        
        # Training data generators
        training_data_df = pd.read_csv(train_df_file)
        num_samples = training_data_df.shape[0]
        steps_per_epoch = num_samples//batch_size
        adversarial_gen = self.create_adversarial_datagen(
            training_data_df, self.img_shape[0], self.img_shape[1],
            batch_size)
        discriminator_valid_gen = self.create_discriminator_valid_datagen(
            training_data_df, self.img_shape[0],
            self.img_shape[1], batch_size)
        discriminator_invalid_gen = self.create_discriminator_invalid_datagen(
            training_data_df, self.img_shape[0],
            self.img_shape[1], batch_size)

        # Keras bug - model is not loaded on graph otherwise
        self.generator.predict(adversarial_gen.__next__()[0])
        #self.discriminator.predict(discriminator_valid_gen.__next__()[0])
        #self.discriminator.predict(discriminator_invalid_gen.__next__()[0])
        #####################################################

        # Train discriminator and adversarial model successively
        for e in range(1, training_epochs+1):
            print('[-] Adversarial model training epoch',
                  '[',e,'/',training_epochs,']',sep='')
            print('[-] Discriminator Valid Data')
            #self.discriminator.fit_generator(discriminator_valid_gen,
            #                                 steps_per_epoch=steps_per_epoch,
            #                                 epochs=e, initial_epoch=e-1,
            #                                 callbacks=callbacks_list,
            #                                 verbose=1)
            print('[-] Discriminator Invalid Data')
            #self.discriminator.fit_generator(discriminator_invalid_gen,
            #                                 steps_per_epoch=steps_per_epoch,
            #                                 epochs=e, initial_epoch=e-1,
            #                                 verbose=1)
            print('[-] Adversarial model')
            self.adversarial.fit_generator(adversarial_gen,
                                           steps_per_epoch=steps_per_epoch,
                                           epochs=e, initial_epoch=e-1,
                                           verbose=1,
                                           use_multiprocessing=True)

            # Save model weights
            if (e % self.args.save_per == 0):
                postfix = '_epoch_' + str(e)
                self.save_model(self.generator, self.model_name,
                                'weights', postfix)

        # Save final weights
        model.save_model(self.generator, self.model_name, 'weights', '_final')


    def __set_weights_trainable(self, model, trainable):
        """
        Sets model weights to trainable/non-trainable.
            Inputs:
                model: Model to set weights.
                trainable: Trainability flag.
        """
        for layer in model.layers:
            layer.trainable = trainable
            if isinstance(layer, Model):
                self.__set_weights_trainable(layer, trainable)
    

    def save_model(self, model, name, mode, postfix=''):
        """
        Saves model architecture or weights in trained_models directory.
            Inputs:
                model: Model to save.
                name: Model name.
                mode: Save either model [weights/architecture].
                postfix: Postfix appended to model name.
        """
        if model is not None:
            # Create saving dir if non-existent
            if not os.path.isdir('trained_models/'):
                os.makedirs('trained_models/')

            ## Keras bug - cannot save nested models if weights are trainable
            self.__set_weights_trainable(model, False)
            ###################################################

            # Save model weights/architecture
            if mode == 'weights':
                filename = ('trained_models/' + name +
                            '_weights' + postfix + '.h5')
                print('[*] Saving model weights at', '"' + filename + '"')
                model.save_weights(filename)
            elif mode == 'architecture':
                filename = 'trained_models/' + name + '.json'
                print('[*] Saving model architecture at', '"' + filename + '"')
                with open(filename, 'w') as write_file:
                    model_architecture_json = model.to_json()
                    json.dump(model_architecture_json, write_file)
            else:
                print('[!] Save mode not supported.')

            ###################################################
            self.__set_weights_trainable(model, True)
            ###################################################

        else:
            print('[!] Model not defined.')


    def __learning_rate_halving(self, start_epoch, period):
        """
        Creates callback function to halve learning rate during training.
            Inputs:
                start_epoch: First epoch to halve learning rate at.
                period: Learning rate halving epoch period.
            Outputs:
                halve_lr: Learning rate halving callback function.
        """
        def halve_lr(epoch, curr_lr):
            # Epochs are zero-indexed
            if (epoch < start_epoch-1):
                return curr_lr
            else:
                if (epoch-(start_epoch-1)) % period == 0:
                    print('[*] Halving learning rate',
                          '(=',curr_lr,' -> ',(curr_lr / 2.0),')', sep='')
                    return curr_lr / 2.0
                else:
                    return curr_lr
        return halve_lr


    def create_discriminator_invalid_datagen(self, training_data_df, height,
            width, batch_size):
        """
        Creates invalid training data generator for the discriminator model.
            Inputs:
                training_data_df: Dataframe with the paths to frame triplets.
                height: Input image height.
                width: Input image width.
                batch_size: Generated batch sizes.
            Outputs:
                ([inputs], [outputs]) tuple for discriminator training.
        """
        # Image preprocessor
        datagen = pre.ImageDataGenerator(
            rescale=1/255,
            channel_shift_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest')

        # Frame generators - use same seed to ensure continuity
        seed = int(np.random.rand(1,1)*2000)
        curr_generator = datagen.flow_from_dataframe(
            training_data_df, directory=None, x_col='curr_frame',
            target_size=(height, width), batch_size=batch_size,
            interpolation='bilinear', class_mode=None, seed=seed)
        prev_generator = datagen.flow_from_dataframe(
            training_data_df, directory=None, x_col='prev_frame',
            target_size=(height,width), batch_size=batch_size,
            interpolation='bilinear', class_mode=None, seed=seed)
        next_generator = datagen.flow_from_dataframe(
            training_data_df, directory=None, x_col='next_frame', 
            target_size=(height,width), batch_size=batch_size,
            interpolation='bilinear', class_mode=None, seed=seed)
    
        while True:
            curr_frame = curr_generator.__next__()
            prev_frame = prev_generator.__next__()
            next_frame = next_generator.__next__()

            generator_samples = self.generator.predict([curr_frame, 
                                                        prev_frame,
                                                        next_frame])
            # Use half reprojections from prev_frame and half from next_frame
            half_batch = curr_frame.shape[0] // 2
            generator_split1 = np.concatenate(
                (generator_samples[0][:half_batch],
                 generator_samples[1][half_batch:]), axis=0)
            generator_split2 = np.concatenate(
                (generator_samples[2][:half_batch],
                 generator_samples[3][half_batch:]), axis=0)
            generator_split3 = np.concatenate(
                (generator_samples[4][:half_batch],
                 generator_samples[5][half_batch:]), axis=0)
            generator_split4 = np.concatenate(
                (generator_samples[6][:half_batch],
                 generator_samples[7][half_batch:]), axis=0)
            validation_labels = np.zeros((generator_samples[0].shape[0], 1))

            # Returns (inputs, outputs) tuple
            yield ([curr_frame,
                    generator_split1, generator_split2,
                    generator_split3, generator_split4],
                   validation_labels)


    def create_discriminator_valid_datagen(self, training_data_df, height,
            width, batch_size):
        """
        Creates valid training data generator for the discriminator model.
            Inputs:
                training_data_df: Dataframe with the paths to frame triplets.
                height: Input image height.
                width: Input image width.
                batch_size: Generated batch sizes.
            Outputs:
                ([inputs], [outputs]) tuple for discriminator training.
        """
        # Image preprocessor
        datagen = pre.ImageDataGenerator(
            rescale=1/255,
            channel_shift_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest')

        # Frame generators - use same seed to ensure continuity
        seed = int(np.random.rand(1,1)*3000)
        curr_generator = datagen.flow_from_dataframe(
            training_data_df, directory=None, x_col='curr_frame',
            target_size=(height, width), batch_size=batch_size,
            interpolation='bilinear', class_mode=None, seed=seed)
    
        while True:
            curr_frame = curr_generator.__next__()
            curr_frame_list = [curr_frame for _ in range(5)]
            validation_labels = np.ones((curr_frame.shape[0],1))

            # Returns (inputs, outputs) tuple
            yield (curr_frame_list, validation_labels)


    def create_adversarial_datagen(self, training_data_df, height, width,
            batch_size):
        """
        Creates tranining data generator for the adversarial model.
            Inputs:
                training_data_df: Dataframe with the paths to frame triplets.
                height: Input image height.
                width: Input image width.
                batch_size: Generated batch sizes.
            Outputs:
                ([inputs], [outputs]) tuple for generator training.
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
            training_data_df, directory=None, x_col='curr_frame',
            target_size=(height,width), batch_size=batch_size,
            interpolation='bilinear', class_mode=None, seed=seed)
        prev_generator = datagen.flow_from_dataframe(
            training_data_df, directory=None, x_col='prev_frame',
            target_size=(height,width), batch_size=batch_size,
            interpolation='bilinear', class_mode=None, seed=seed)
        next_generator = datagen.flow_from_dataframe(
            training_data_df, directory=None, x_col='next_frame',
            target_size=(height,width), batch_size=batch_size,
            interpolation='bilinear', class_mode=None, seed=seed)
    
        while True:
            curr_frame = curr_generator.__next__()
            prev_frame = prev_generator.__next__()
            next_frame = next_generator.__next__()

            # Returns (inputs,outputs) tuple
            batch_size_curr = curr_frame.shape[0]
            curr_frame_list = [curr_frame for _ in range(4)]
            # Downsample curr frame
            curr_frame_scales = []
            for s in [1, 2, 4, 8]:
                curr_frame_scales += [curr_frame[:,::s,::s,:]]
            #true_labels = [np.ones((batch_size_curr, 1)) for _ in range(2)]

            yield ([curr_frame, prev_frame, next_frame], 
                   (curr_frame_list + curr_frame_scales))



if __name__ == '__main__':
    model = MonocularAdversarialModel()

    if model.args.train == True:
        model.initialize_training()
        model.train(model.args.train_dataframe, model.args.batch_size,
                    model.args.training_epochs)
    else:
        print('[!] Unknown operation, use -h flag for help.')

