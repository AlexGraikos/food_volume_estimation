import os
import argparse
import numpy as np
import pandas as pd
import json
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, LambdaCallback
import keras.preprocessing.image as pre
from custom_modules import *
from networks import NetworkBuilder


class MonovideoModel:
    def __init__(self):
        """"
        Reads command-line arguments and initializes general model parameters.
        """
        self.args = self.__parse_args()
        # Load training parameters
        self.model_name = self.args.model_name
        with open(self.args.config, 'r') as read_file:
            config = json.load(read_file)
            self.img_shape = tuple(config['img_size'])
            self.intrinsics_mat = np.array(config['intrinsics'])
            self.depth_range = config['depth_range']
            self.dataset = config['name']
        print('[*] Training model on', self.dataset, 'dataset.')
        print('[*] Input image size:', self.img_shape)
        print('[*] Predicted depth range:', self.depth_range)


    def __parse_args(self):
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
                            help='Training dataFrame file (.csv).',
                            default=None)
        parser.add_argument('--config', type=str, 
                            help='Dataset configuration file (.json).',
                            default=None)
        parser.add_argument('--starting_weights', type=str,
                            help='Starting weights file (.h5).',
                            default=None)
        parser.add_argument('--batch_size', type=int, 
                            help='Training batch size.',
                            default=8)
        parser.add_argument('--training_epochs', type=int, 
                            help='Model training epochs.',
                            default=10)
        parser.add_argument('--model_name', type=str, 
                            help='Model name, used during saving.',
                            default='monovideo')
        parser.add_argument('--save_per', type=int, 
                            help='Epochs between model saving.',
                            default=5)
        args = parser.parse_args()
        return args


    def initialize_training(self):
        """
        Initializes model training.
        """
        # Create monovideo model
        nets_builder = NetworkBuilder(self.img_shape, self.intrinsics_mat,
                                      self.depth_range)
        self.monovideo = nets_builder.create_monovideo()
        self.save_model(self.monovideo, self.model_name, 'architecture')
        print('[*] Created monovideo model.')
        if self.args.starting_weights is not None:
            ## Keras bug - cannot load nested models if weights are trainable
            self.__set_weights_trainable(self.monovideo, False)
            ###################################################
            self.monovideo.load_weights(self.args.starting_weights)
            print('[*] Initialized weights from:', self.args.starting_weights)
            ## Keras bug - cannot load nested models if weights are trainable
            self.__set_weights_trainable(self.monovideo, True)
            ###################################################

        # Synthesize training model
        augmented_inputs = self.monovideo.output[0:3]
        reprojections = self.monovideo.outputs[3:11]
        per_scale_reprojections = self.monovideo.outputs[11:15]
        inverse_depths = self.monovideo.outputs[15:19]
        depth_maps = self.monovideo.outputs[19:23]
        self.training_model = Model(inputs=self.monovideo.input,
                                    outputs=(per_scale_reprojections
                                             + inverse_depths),
                                    name='training_model')
        print('[*] Created training model.')

        # Compile
        adam_opt = Adam(lr=1e-4)
        custom_losses = Losses()
        loss_list = ([custom_losses.reprojection_loss() for s in range(4)] 
                     + [custom_losses.depth_smoothness() for s in range(4)])
        loss_weights = ([1 for s in range(4)]
                        + [(0.01 / (2 ** s)) for s in range(4)])
        self.training_model.compile(adam_opt, loss=loss_list,
                                 loss_weights=loss_weights)


    def train(self, train_df_file, batch_size, training_epochs):
        """
        Trains model.
            Inputs:
                train_df_file: Training data dataFrame file (.csv).
                batch_size: Training batch size.
                training_epochs: Number of training epochs.
        """
        # Learning rate dropping callback
        lr_callback = self.__learning_rate_dropping(
            factor=2, start_epoch=15, period=6)
        checkpoint_callback = self.__model_checkpoint
        callbacks_list = [LearningRateScheduler(schedule=lr_callback,
                                                verbose=0),
                          LambdaCallback(on_epoch_end=checkpoint_callback)]
        
        # Training data generator
        training_data_df = pd.read_csv(train_df_file)
        train_data_generator = DataGenerator(
            training_data_df, self.img_shape[0], self.img_shape[1],
            batch_size, True)

        # Train model
        training_history = self.training_model.fit_generator(
            train_data_generator, epochs=training_epochs, verbose=1,
            callbacks=callbacks_list)

        # Save final weights and training history
        self.save_model(self.monovideo, self.model_name, 'weights', '_final')
        with open('trained_models/training_history.json', 'w') as log_file:
            json.dump(training_history.history, log_file, indent=4,
                      cls=NumpyEncoder)
            print('[*] Saving training log at', 
                  '"trained_models/training_history.json".')


    def save_model(self, model, name, mode, postfix=''):
        """
        Saves model architecture/weights in trained_models directory.
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
                print('[*] Saving model weights at', '"' + filename + '".')
                model.save_weights(filename)
            elif mode == 'architecture':
                filename = 'trained_models/' + name + '.json'
                print('[*] Saving model architecture at',
                      '"' + filename + '".')
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


    def __model_checkpoint(self, epoch, logs):
        """
        Callback function that saves model per given period of epochs.
            Inputs:
                epoch: Current epoch (zero-indexed).
                logs: Training logs.
        """
        if (epoch + 1) % self.args.save_per == 0:
            postfix = '_epoch_' + '{}'.format(epoch + 1)
            self.save_model(self.monovideo, self.model_name, 'weights',
                            postfix)


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
    

    def __learning_rate_dropping(self, factor, start_epoch, period):
        """
        Creates callback function to reduce learning rate by given factor
        during training.
            Inputs:
                factor: Learning rate reduction factor.
                start_epoch: First epoch to reduce learning rate at.
                period: Learning rate reduction epoch period.
            Outputs:
                drop_lr: Learning rate dropping callback function.
        """
        def drop_lr(epoch, curr_lr):
            # Epochs are zero-indexed
            if (epoch < start_epoch - 1):
                return curr_lr
            else:
                if (epoch - (start_epoch - 1)) % period == 0:
                    print('[*] Reducing learning rate /', factor,
                          ' (', curr_lr, ' -> ', (curr_lr / factor), ').',
                          sep='')
                    return curr_lr / factor
                else:
                    return curr_lr
        return drop_lr


if __name__ == '__main__':
    model = MonovideoModel()

    if model.args.train == True:
        model.initialize_training()
        model.train(model.args.train_dataframe, model.args.batch_size,
                    model.args.training_epochs)
    else:
        print('[!] Unknown operation, use -h flag for help.')

