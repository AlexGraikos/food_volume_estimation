import argparse


TRAIN_ANNOTATIONS_PATH = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_training_set_release_2.0/annotations.json"
TRAIN_IMAGE_DIRECTORY = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_training_set_release_2.0/images/"

VAL_ANNOTATIONS_PATH = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_validation_set_2.0/annotations.json"
VAL_IMAGE_DIRECTORY = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_validation_set_2.0/images/"

class FoodRecognitionOptions:

    def __init__(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-b", "--batch_size", type=int, default=12, help="Defines the training batch size")
        parser.add_argument("-s", "--steps_epoch", type=int, default=1000, help="Defines the training steps per epoch")
        parser.add_argument("-e", "--epochs", type=int, default=50, help="Defines the training number of epochs")
        parser.add_argument("-ta", "--train_ann", 
                            type=str, 
                            default=TRAIN_ANNOTATIONS_PATH,
                            help="Path to training annotations")
        parser.add_argument("-ti", "--train_img", 
                            type=str, 
                            default=TRAIN_IMAGE_DIRECTORY,
                            help="Path to training image dir")
        parser.add_argument("-va", "--val_ann", 
                            type=str, 
                            default=VAL_ANNOTATIONS_PATH,
                            help="Path to validation annotations")
        parser.add_argument("-vi", "--val_img", 
                            type=str, 
                            default=VAL_IMAGE_DIRECTORY,
                            help="Path to training image dir")
        parser.add_argument("-d", "--data_size", 
                            type=tuple, 
                            default=(224,224), 
                            help="The generator output size")
        args = parser.parse_args()

        self.batch_size = args.batch_size
        self.steps_epoch = args.steps_epoch
        self.epochs = args.epochs
        self.data_size = args.data_size

        self.train_ann_path = args.train_ann
        self.val_ann_path = args.val_ann
        self.train_img_path = args.train_img
        self.val_img_path = args.val_img
        