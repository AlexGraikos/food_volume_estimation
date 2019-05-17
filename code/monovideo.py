import os
import argparse
import numpy as np
import pandas as pd
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, LambdaCallback
import keras.preprocessing.image as pre
import json
from custom_modules import *
from networks import Networks


class MonovideoModel:
    def __init__(self):
        """"
        Reads command-line arguments and initializes general model parameters.
        """
        self.args = self.__parse_args()
        # Model parameters
        self.img_shape = (self.args.img_height, self.args.img_width, 3)
        self.model_name = self.args.model_name


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
        Initializes model training.
        """
        nets = Networks(self.img_shape)
        # Create monovideo model
        self.monovideo = nets.create_full_model()
        print('[*] Created model')
        self.save_model(self.monovideo, self.model_name, 'architecture')

        # Synthesize training model
        augmented_inputs = self.monovideo.output[0:3]
        reprojections = self.monovideo.outputs[3:11]
        inverse_depths = self.monovideo.outputs[11:15]
        per_scale_reprojections = self.monovideo.outputs[15:19]
        self.training_model = Model(inputs=self.monovideo.input,
                                    outputs=(per_scale_reprojections
                                             + inverse_depths),
                                    name='training_model')
        print('[*] Created training model')

        # Compile
        adam_opt = Adam(lr=1e-4)
        custom_losses = Losses()
        loss_list = ([custom_losses.reprojection_loss() for s in range(4)] 
                     + [custom_losses.depth_smoothness() for s in range(4)])
        loss_weights = ([1 for s in range(4)]
                        + [(0.001 / (2 ** s)) for s in range(4)])
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
        # Learning rate halving callback
        lr_callback = self.__learning_rate_halving(start_epoch=10,
                                                   period=5)
        checkpoint_callback = self.__model_checkpoint
        callbacks_list = [LearningRateScheduler(schedule=lr_callback,
                                                verbose=0),
                          LambdaCallback(on_epoch_end=checkpoint_callback)]
        
        # Training data generator
        training_data_df = pd.read_csv(train_df_file)
        num_samples = training_data_df.shape[0]
        steps_per_epoch = num_samples//batch_size
        training_data_gen = self.create_training_data_gen(
            training_data_df, self.img_shape[0], self.img_shape[1],
            batch_size)

        # Train model
        training_history = self.training_model.fit_generator(
            training_data_gen, steps_per_epoch=steps_per_epoch,
            epochs=training_epochs, verbose=1, callbacks=callbacks_list)

        # Save final weights and training history
        model.save_model(self.monovideo, self.model_name, 'weights', '_final')
        with open('trained_models/training_history.json', 'w') as log_file:
            json.dump(training_history.history, log_file, indent=4,
                      cls=NumpyEncoder)
            print('[*] Saving training log at', 
                  '"trained_models/training_history.json"')
    

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


    def __model_checkpoint(self, epoch, logs):
        """
        Callback function that saves model per given period of epochs.
            Inputs:
                epoch: Current epoch (zero-indexed).
                logs: Training logs.
        """
        if (epoch + 1) % self.args.save_per == 0:
            postfix = '_epoch_' + str(epoch + 1)
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


    def create_training_data_gen(self, training_data_df, height, width,
            batch_size):
        """
        Creates tranining data generator for the training model.
            Inputs:
                training_data_df: Dataframe with the paths to frame triplets.
                height: Input image height.
                width: Input image width.
                batch_size: Generated batch sizes.
            Outputs:
                ([inputs], [outputs]) tuple for model training.
        """
        # Image preprocessor. Horizontal flipping is applied to both
        # inputs and target outputs
        datagen = pre.ImageDataGenerator(rescale=1/255,
                                         horizontal_flip=True,
                                         fill_mode='nearest')

        # Frame generators - use same seed to ensure continuity
        seed = int(np.random.rand(1)*1000)
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

            yield ([curr_frame, prev_frame, next_frame], 
                   (curr_frame_list + curr_frame_scales))



if __name__ == '__main__':
    model = MonovideoModel()

    if model.args.train == True:
        model.initialize_training()
        model.train(model.args.train_dataframe, model.args.batch_size,
                    model.args.training_epochs)
    else:
        print('[!] Unknown operation, use -h flag for help.')

