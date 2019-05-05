from __future__ import print_function
import os, shutil, argparse
import numpy as np
import pandas as pd
import cv2

class DataUtils():

    def __init__(self):
        self.args = self.parse_args()


    def parse_args(self):
        """
        Parses command-line input arguments
            Outputs:
                args: The arguments object
        """
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Data processing utilities.')
        parser.add_argument('--createSet', action='store_true', help='Create image set from sources.', default=False)
        parser.add_argument('--createSetDF', action='store_true', help='Create image set dataFrame from sources.', default=False)
        parser.add_argument('--data_source_file', type=str, help='File containing the data sources.', default=None)
        parser.add_argument('--save_target', type=str, help='Target directory/file to save created set/dataFrame.', default=None)
        parser.add_argument('--target_width', type=int, help='Target image width.', default=224)
        parser.add_argument('--target_height', type=int, help='Target image height.', default=128)
        parser.add_argument('--interp', type=str, help='Interpolation method [nearest/bilinear/cubic].', default='nearest')
        parser.add_argument('--stride', type=int, help='Frame triplet stride.', default=1)
        args = parser.parse_args()
        return args


    def createSequenceDataFrame(self, img_directory, stride=1):
        """
        Creates a frame sequence dataFrame from images in given directory
        The created dataFrame has columns [curr_frame, prev_frame, next_frame]
            Inputs:
                img_directory: The directory for which to create the sequence dataframe
                stride: Distance between frames [prev <-> curr <-> next]
            Outputs:
                sequence_df: The image sequence dataFrame
        """
        image_files = [os.path.join(img_directory,f) for f in os.listdir(img_directory)
            if os.path.isfile(os.path.join(img_directory,f))]
        image_files.sort() # Sort in sequential order

        # Create sequential frame triplet dataframe
        curr_frames = image_files[stride:-stride:stride]
        prev_frames = image_files[:-2*stride:stride]
        next_frames = image_files[2*stride::stride]
        frame_triplets = np.transpose(np.array([curr_frames, prev_frames, next_frames]))
        sequence_df = pd.DataFrame(data=frame_triplets, columns=['curr_frame', 'prev_frame', 'next_frame'])
        return sequence_df


    def createSetDataFrame(self, data_source_file, df_file, stride=1):
        """
        Creates a sequence dataFrame that describes a set of images defind in data_source_file
        The dataFrame is saved in df_file (HDF5 format, key=df)
        The created dataFrame has columns [curr_frame, prev_frame, next_frame]
            Inputs:
                data_source_file: File defining image directories that compose the target set
                df_file: The target file to save the dataFrame (.h5)
                stride: Distance between frames [prev <-> curr <-> next]
        """
        # Load data source directories
        data_sources = [line.rstrip('\n') for line in open(data_source_file, 'r')]
        data_sources = [l for l in data_sources if l != ''] # Remove empty lines

        # For each data source directory create its sequence dataFrame
        print('[*] Using stride',stride)
        sequences = []
        for source in data_sources:
            print('[*] Generating sequence from', source)
            sequence_df = self.createSequenceDataFrame(source, stride)
            sequences.append(sequence_df)

        # Concat all and save
        set_df = pd.concat(sequences, axis=0, ignore_index=True)
        set_df.to_hdf(df_file, key='df', format='fixed', mode='w')
        print('[*] Total frames:', set_df.shape[0])
        print('[*] Finished creating set dataFrame and saved in', '"'+os.path.abspath(df_file)+'"')
        

    def createSet(self, data_source_file, target_dir, target_size=None, interpolation='nearest'):
        """
        Creates a set of images from source directories defined in data_source_file 
        The set is saved at target_dir with target_size image sizes
            Inputs:
                data_source_file: File defining image directories that compose the target set
                target_dir: The target directory to save the set
                target_size: Target image size
                interpolation: Interpolation method [nearest (default)/bilinear/cubic]
        """
        # Load data source directories
        data_sources = [line.rstrip('\n') for line in open(data_source_file, 'r')]
        data_sources = [l for l in data_sources if l != ''] # Remove empty lines

        # Create target directory
        try:
            os.mkdir(target_dir)
        except Exception:
            print('[!] Target directory already exists')
            r = input('[!] Clean up directory "' + target_dir + '" ? [y/n]: ')
            if r == 'y' or r == 'Y':
                # Cleanup target directory
                shutil.rmtree(target_dir)
                os.mkdir(target_dir)
            else:
                print('[*] Exiting...')
                exit()

        # Image interpolation method
        if interpolation == 'nearest':
            print('[*] Using nearest neighbor interpolation')
            cv2_interp = cv2.INTER_NEAREST
        elif interpolation == 'bilinear':
            print('[*] Using bilinear interpolation')
            cv2_interp = cv2.INTER_LINEAR
        elif interpolation == 'cubic':
            print('[*] Using bicubic interpolation')
            cv2_interp = cv2.INTER_CUBIC
        else:
            print('[!] Unknown interpolation method, using nearest neighbor')
            cv2_interp = cv2.INTER_NEAREST

        # For each data source directory load images, resize, and save in target directory
        source_files = 0
        target_images = 0
        for source in data_sources:
            print('[*] Loading data from', source)
            image_files = [f for f in os.listdir(source)
                if os.path.isfile(os.path.join(source,f))]
            source_files += len(os.listdir(source))

            for i in image_files:
                source_file_path = os.path.join(source,i)
                # Compose target file name
                prefix = os.path.basename(os.path.dirname(source_file_path)) + '_'
                target_file_path = os.path.join(target_dir, prefix + i)

                img = cv2.imread(source_file_path, cv2.IMREAD_COLOR) 
                if target_size is None:
                    target_size = img.shape[:2]
                img_resized = cv2.resize(img, target_size[::-1], interpolation=cv2_interp)
                cv2.imwrite(target_file_path, img_resized)
                target_images += 1

        # Check number of source and target files
        if source_files != target_images:
            print('[!] Mismatch between number of source and target images')
        print('[*] Finished creating set')
            


if __name__ == '__main__':
    imgUtils = DataUtils()

    if imgUtils.args.createSet == True:
        imgUtils.createSet(imgUtils.args.data_source_file, imgUtils.args.save_target,
            target_size=(imgUtils.args.target_height, imgUtils.args.target_width),
            interpolation=imgUtils.args.interp)
    elif imgUtils.args.createSetDF == True:
        imgUtils.createSetDataFrame(imgUtils.args.data_source_file, imgUtils.args.save_target,
        stride=imgUtils.args.stride)
    else:
        print('[!] Unknown operation argument, use -h flag for help.')

