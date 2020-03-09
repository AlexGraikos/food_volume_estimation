from __future__ import print_function
import os, shutil, argparse
import numpy as np
import pandas as pd
import cv2


class DataUtils():
    def __init__(self):
        self.args = self.parse_args()
        self.FLOW_THRESHOLD = 1

    def parse_args(self):
        """Parses command-line input arguments.

        Outputs:
            args: The arguments object.
        """
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description='Data processing utilities.')
        parser.add_argument('--create_EPIC_set', action='store_true',
                            help=('Create image set from '
                                  + 'EPIC-kitchens dataset sources.'),
                            default=False)
        parser.add_argument('--create_set_df', action='store_true',
                            help='Create sequence dataFrame from sources.',
                            default=False)
        parser.add_argument('--create_dir_df', action='store_true', 
                            help='Create sequence dataFrame from directory.', 
                            default=False)
        parser.add_argument('--data_source', type=str,
                            help='File/Directory containing the source data.',
                            default=None)
        parser.add_argument('--save_target', type=str, 
                            help=('Target directory/file to save created ' +
                                  'set/dataFrame.'),
                            default=None)
        parser.add_argument('--target_width', type=int,
                            help='Target image width.', 
                            default=224)
        parser.add_argument('--target_height', type=int, 
                            help='Target image height.', 
                            default=128)
        parser.add_argument('--interp', type=str,
                            help=('Interpolation method ' +
                                  '[nearest/bilinear/cubic].'),
                            default='nearest')
        parser.add_argument('--stride', type=int, 
                            help='Frame triplet creation stride.', 
                            default=1)
        args = parser.parse_args()
        return args

    def create_directory_dataframe(self, directory, df_file, stride=1):
        """Creates a sequence dataFrame from a directory of frames.
        The dataFrame is saved in df_file (csv format).
        The created dataFrame has columns [curr_frame, prev_frame, next_frame].

        Inputs:
            directory: Directory containing frames composing
                       the target set.
            df_file: The target file to save the dataFrame (.csv).
        """

        # Concat all and save
        directory_df = self.__create_sequence_dataframe(directory, stride)
        directory_df.to_csv(df_file, index=False)
        print('[*] Total frames:', directory_df.shape[0])
        print('[*] Finished creating set dataFrame and saved in',
              '"' + os.path.abspath(df_file) + '"')
 
    def create_set_dataframe(self, data_source_file, df_file, stride=1):
        """Creates a sequence dataFrame that describes a set of images
        defined in data_source_file.
        The dataFrame is saved in df_file (csv format).
        The created dataFrame has columns [curr_frame, prev_frame, next_frame].

        Inputs:
            data_source_file: File defining image directories that 
                              compose the target set.
            df_file: The target file to save the dataFrame (.csv).
            stride: Distance between frames.
                    [prev <-stride-> curr <-stride-> next]
        """
        # Load data source directories
        data_sources = [line.rstrip('\n') 
                        for line in open(data_source_file, 'r')]
        # Remove empty lines
        data_sources = [l for l in data_sources if l != '']

        # For each data source directory create its sequence dataFrame
        print('[*] Using stride',stride)
        sequences = []
        for source in data_sources:
            print('[*] Generating sequence from', source)
            sequence_df = self.__create_sequence_dataframe(source, stride)
            sequences.append(sequence_df)

        # Concat all and save
        set_df = pd.concat(sequences, axis=0, ignore_index=True)
        set_df.to_csv(df_file, index=False)
        print('[*] Total frames:', set_df.shape[0])
        print('[*] Finished creating set dataFrame and saved in',
              '"' + os.path.abspath(df_file) + '"')

    def __create_sequence_dataframe(self, img_directory, stride=1):
        """Creates a frame sequence dataFrame from images in given directory.
        The created dataFrame has columns [curr_frame, prev_frame, next_frame].

        Inputs:
            img_directory: The directory for which to create 
                           the sequence dataframe.
            stride: Distance between frames.
                    [prev <-stride-> curr <-stride-> next]
        Outputs:
            sequence_df: The image sequence dataFrame.
        """
        image_files = [os.path.join(img_directory, f) 
                       for f in os.listdir(img_directory)
                       if os.path.isfile(os.path.join(img_directory, f))]
        image_files.sort() # Sort in temporal order
        image_files = list(map(os.path.abspath, image_files))

        # Create sequential frame triplet dataframe
        curr_frames = image_files[stride:-stride:stride]
        prev_frames = image_files[:-2*stride:stride]
        next_frames = image_files[2*stride::stride]
        frame_triplets = np.transpose(
            np.array([curr_frames, prev_frames, next_frames]))
        sequence_df = pd.DataFrame(data=frame_triplets,
                                   columns=['curr_frame',
                                            'prev_frame',
                                            'next_frame'])
        return sequence_df

    def create_EPIC_set(self, data_source_file, target_dir, target_size,
            stride, interpolation):
        """Creates a set of EPIC-kitchens dataset images from source
        directories defined in data_source_file.
        The set is filtered by ignoring images with mean optical flow < 1.
        The set is saved at target_dir with target_size image sizes.

        Inputs:
            data_source_file: File defining image directories 
                              that compose the target set.
            target_dir: The target directory to save the set.
            target_size: Target image size.
            stride: Frame loading stride. For EPIC-kitchens the default=2.
            interpolation: Interpolation method.
                           [nearest (default)/bilinear/cubic]
        """
        # Load data source directories
        data_sources = [line.rstrip('\n') 
                        for line in open(data_source_file, 'r')]
        # Remove empty lines
        data_sources = [l for l in data_sources if l != '']

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

        # For each data source directory load images, resize 
        # and save in target directory
        total_frames = 0
        target_frames = 0
        rejected_frames = 0
        failed_frames = 0
        source_index = 0
        for source_rgb in data_sources:
            # Generate flow filepath
            dirs_list = os.path.abspath(source_rgb).split('/')
            dirs_list[dirs_list.index('rgb')] = 'flow' # TODO: use only the "last" rgb directory
            source_flow = '/'.join(dirs_list)

            print('[*] Loading data from', source_rgb)
            image_files = [f for f in os.listdir(source_rgb)
                if os.path.isfile(os.path.join(source_rgb,f))]
            total_frames += len(image_files)
            flow_counter = 0
            # For each frame load its rgb and flow images
            for rgb_counter in range(1, len(image_files)+1, stride):
                flow_counter += 1

                frame_name_rgb = 'frame_{:010d}.jpg'.format(rgb_counter)
                frame_name_flow = 'frame_{:010d}.jpg'.format(flow_counter)
                rgb_file_path = os.path.join(source_rgb, frame_name_rgb)
                flow_u_file_path = os.path.join(source_flow + '/u/', 
                                                frame_name_flow)
                flow_v_file_path = os.path.join(source_flow + '/v/',
                                                frame_name_flow)
                img = cv2.imread(rgb_file_path, cv2.IMREAD_COLOR) 
                flow_u = cv2.imread(flow_u_file_path, cv2.IMREAD_UNCHANGED)
                flow_v = cv2.imread(flow_u_file_path, cv2.IMREAD_UNCHANGED)
                # If any image loading failed continue to next
                if flow_u is None or flow_v is None or img is None:
                    failed_frames += 1
                    continue
                # Compute optical flow mean and ignore if < 1 pixel
                flow_u_scaled = flow_u*(50/255) - 25
                flow_v_scaled = flow_v*(50/255) - 25
                flow_mean = (np.mean(np.sqrt(np.square(flow_u_scaled)
                             + np.square(flow_v_scaled))))
                if flow_mean < self.FLOW_THRESHOLD:
                    rejected_frames += 1
                    continue

                # Compose target file name and save resized image
                prefix = 'source_' + str(source_index) + '_'
                target_file_path = os.path.join(target_dir,
                                                prefix + frame_name_rgb)
                # If none provided, use input image shape
                if target_size is None:
                    target_size = img.shape[:2]
                img_resized = cv2.resize(img, target_size[::-1], 
                                         interpolation=cv2_interp)
                cv2.imwrite(target_file_path, img_resized)
                target_frames += 1

            source_index += 1

        print('[*] Finished creating set')
        print('[*] Total frames: %d' % total_frames,
              'Target frames: %d' % target_frames, 
              'Rejected frames: %d' % rejected_frames,
              'Failed frames: %d' % failed_frames)



if __name__ == '__main__':
    imgUtils = DataUtils()

    if imgUtils.args.create_EPIC_set == True:
        imgUtils.create_EPIC_set(
            imgUtils.args.data_source, imgUtils.args.save_target,
            target_size=(imgUtils.args.target_height,
                         imgUtils.args.target_width),
            stride=imgUtils.args.stride, interpolation=imgUtils.args.interp)
    elif imgUtils.args.create_set_df == True:
        imgUtils.create_set_dataframe(
            imgUtils.args.data_source, imgUtils.args.save_target,
            stride=imgUtils.args.stride)
    elif imgUtils.args.create_dir_df == True:
        imgUtils.create_directory_dataframe(
            imgUtils.args.data_source, imgUtils.args.save_target,
            imgUtils.args.stride)
    else:
        print('[!] Unknown operation, use -h flag for help.')

