import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D, GlobalAveragePooling2D, \
    LeakyReLU, Dense, Flatten, Input, Concatenate, Lambda, BatchNormalization
from classification_models.keras import Classifiers
from food_volume_estimation.depth_estimation.custom_modules import *

ResNet18, _ = Classifiers.get('resnet18')

class NetworkBuilder:
    def __init__(self, img_shape, intrinsics_matrix=None,
            depth_range=[0.01, 10]):
        # Model parameters
        self.img_shape = img_shape
        self.intrinsics_matrix = intrinsics_matrix
        if intrinsics_matrix is not None:
            # Scale intrinsics matrix to image size (in pixel dims)
            x_scaling = self.img_shape[1] / 1920
            y_scaling = self.img_shape[0] / 1080
            self.intrinsics_matrix[0, :] *= x_scaling
            self.intrinsics_matrix[1, :] *= y_scaling
        self.depth_range = depth_range
        self.custom_losses = Losses()

    def create_monovideo(self):
        """Creates the complete monocular depth estimation model.

        Outputs:
            monovideo: Monocular depth estimation model, with all
            available outputs.
        """
        # Create modules
        depth_net = self.__create_depth_net()
        pose_net = self.__create_pose_net()
        reprojection_module = self.__create_reprojection_module()

        # Inputs
        curr_frame = Input(shape=self.img_shape)
        prev_frame = Input(shape=self.img_shape)
        next_frame = Input(shape=self.img_shape)
        # Augmented inputs (not used in reprojection module)
        augmented_inputs = AugmentationLayer()([curr_frame, prev_frame,
                                                next_frame])
        curr_frame_aug, prev_frame_aug, next_frame_aug = augmented_inputs
        # Depth
        inverse_depths = depth_net(curr_frame_aug)
        # Poses
        pose_prev = pose_net([prev_frame_aug, curr_frame_aug])
        pose_next = pose_net([next_frame_aug, curr_frame_aug])
        # Reprojections and depth maps
        module_outputs = reprojection_module(
            [prev_frame, next_frame, pose_prev, pose_next] + inverse_depths)
        reprojections = module_outputs[:8]
        depth_maps = module_outputs[8:]
        # Inputs and per-scale reprojections for automasking and 
        # per-scale min loss
        source_loss = Lambda(
            self.custom_losses.compute_source_loss,
            arguments={'alpha': 0.85})(
            [curr_frame, prev_frame, next_frame])
        per_scale_reprojections = [
            Concatenate(name='scale1_reprojections')([source_loss,
                                                      reprojections[0],
                                                      reprojections[1]]),
            Concatenate(name='scale2_reprojections')([source_loss,
                                                      reprojections[2],
                                                      reprojections[3]]),
            Concatenate(name='scale3_reprojections')([source_loss,
                                                      reprojections[4],
                                                      reprojections[5]]),
            Concatenate(name='scale4_reprojections')([source_loss,
                                                      reprojections[6],
                                                      reprojections[7]])]
        monovideo = Model(
            inputs=[curr_frame, prev_frame, next_frame],
            outputs=(augmented_inputs
                     + reprojections + per_scale_reprojections
                     + inverse_depths + depth_maps),
            name='monovideo')
        return monovideo 

    def __create_depth_net(self):
        """Creates the depth predicting network model.

        Outputs:
            depth_net: Depth predicting network model.
        """
        # ResNet18 encoder
        depth_encoder = ResNet18(input_shape=self.img_shape,
                                 weights='imagenet', include_top=False)
        skip1 = depth_encoder.get_layer('relu0').output
        skip2 = depth_encoder.get_layer('stage2_unit1_relu1').output
        skip3 = depth_encoder.get_layer('stage3_unit1_relu1').output
        skip4 = depth_encoder.get_layer('stage4_unit1_relu1').output

        # Decoder
        def dec_layer(prev_layer, skip_layer, filters, upsample):
            dec = prev_layer
            if upsample:
                dec = UpSampling2D(size=(2,2))(dec)
            if skip_layer is not None:
                dec = Concatenate()([skip_layer, dec])
            dec = ReflectionPadding2D(padding=(1,1))(dec)
            dec = Conv2D(filters=filters, kernel_size=3,
                         activation='elu')(dec)
            return dec

        def inverse_depth_layer(prev_layer):
            inverse_depth = ReflectionPadding2D(padding=(1,1))(prev_layer)
            inverse_depth = Conv2D(filters=1, kernel_size=3, 
                                   activation='sigmoid')(inverse_depth)
            return inverse_depth
        # Layers
        upconv5 = dec_layer(depth_encoder.output, None, 256, False)
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
        inverse_depth_4 = inverse_depth_layer(iconv4)
        inverse_depth_3 = inverse_depth_layer(iconv3)
        inverse_depth_2 = inverse_depth_layer(iconv2)
        inverse_depth_1 = inverse_depth_layer(iconv1)
        # Model
        depth_net = Model(
            inputs=depth_encoder.input,
            outputs=[inverse_depth_1, inverse_depth_2,
                     inverse_depth_3, inverse_depth_4],
            name='depth_net')
        return depth_net

    def __create_pose_net(self):
        """Creates and returns the pose estimation network.

        Outputs:
            pose_net: Pose estimation network model.
        """
        # Pose encoder
        pose_encoder = ResNet18(input_shape=(self.img_shape[0],
                                             self.img_shape[1], 6),
                                weights=None, include_top=False)

        # Copy pre-trained ResNet18 weights to 6-channel pose ResNet encoder
        pose_encoder_weights_source = ResNet18(
            input_shape=self.img_shape, weights='imagenet', include_top=False)
        weights = [l.get_weights() 
                   for l in pose_encoder_weights_source.layers]
        for i in range(1, len(pose_encoder.layers)):
            if i == 1:
                # Tile batchnorm weights and copy
                bn_weights = list(map(lambda x: np.tile(x, 2), weights[i]))
                pose_encoder.layers[i].set_weights(bn_weights)
            elif i == 3:
                # Tile conv weights along channel axis and copy
                conv_weights = list(map(lambda x: np.tile(x, (1, 1, 2, 1)),
                                    weights[i]))
                pose_encoder.layers[i].set_weights(conv_weights)
            else:
                # Rest of weights match between the two models
                layer_weights = weights[i]
                pose_encoder.layers[i].set_weights(layer_weights)
        
        # Inputs
        source_frame = Input(shape=self.img_shape)
        target_frame = Input(shape=self.img_shape)
        concatenated_frames = Concatenate()([source_frame, target_frame])
        # Pose net
        pose_features = pose_encoder(concatenated_frames)
        pconv0 = Conv2D(filters=256, kernel_size=1, padding='same',
                        activation='relu')(pose_features)
        pconv1 = Conv2D(filters=256, kernel_size=3, padding='same',
                        activation='relu')(pconv0)
        pconv2 = Conv2D(filters=256, kernel_size=3, padding='same',
                        activation='relu')(pconv1)
        pconv3 = Conv2D(filters=6, kernel_size=1, padding='same',
                        activation='linear')(pconv2)
        pose = GlobalAveragePooling2D()(pconv3)
        # Model
        pose_net = Model(
            inputs=[source_frame, target_frame],
            outputs=pose, 
            name='pose_net')
        return pose_net

    def __create_reprojection_module(self):
        """Creates and returns the reprojection module model.

        Outputs:
            reprojection_module: Reprojection module model.
        """
        # Inputs
        prev_frame = Input(shape=self.img_shape)
        next_frame = Input(shape=self.img_shape)
        pose_prev = Input(shape=(6,))
        pose_next = Input(shape=(6,))
        inverse_depth_1 = Input(shape=(self.img_shape[0],
                                       self.img_shape[1], 1))
        inverse_depth_2 = Input(shape=(self.img_shape[0] // 2,
                                       self.img_shape[1] // 2, 1))
        inverse_depth_3 = Input(shape=(self.img_shape[0] // 4,
                                       self.img_shape[1] // 4, 1))
        inverse_depth_4 = Input(shape=(self.img_shape[0] // 8,
                                       self.img_shape[1] // 8, 1))
        # Upsample and normalize inverse depth maps
        inverse_depth_2_up = UpSampling2D(size=(2,2))(inverse_depth_2)
        inverse_depth_3_up = UpSampling2D(size=(4,4))(inverse_depth_3)
        inverse_depth_4_up = UpSampling2D(size=(8,8))(inverse_depth_4)
        depth_map_1 = InverseDepthNormalization(
            self.depth_range[0], self.depth_range[1])(inverse_depth_1)
        depth_map_2 = InverseDepthNormalization(
            self.depth_range[0], self.depth_range[1])(inverse_depth_2_up)
        depth_map_3 = InverseDepthNormalization(
            self.depth_range[0], self.depth_range[1])(inverse_depth_3_up)
        depth_map_4 = InverseDepthNormalization(
            self.depth_range[0], self.depth_range[1])(inverse_depth_4_up)
        depth_maps = [depth_map_1, depth_map_2, depth_map_3, depth_map_4]

        # Create reprojections for each depth map scale from highest to lowest
        reprojections = []
        for depth in depth_maps:
            prev_to_target = ProjectionLayer(
                self.intrinsics_matrix)([prev_frame, depth, pose_prev])
            next_to_target = ProjectionLayer(
                self.intrinsics_matrix)([next_frame, depth, pose_next])
            reprojections += [prev_to_target, next_to_target]

        # Model
        reprojection_module = Model(inputs=[prev_frame, next_frame,
                                            pose_prev, pose_next,
                                            inverse_depth_1, inverse_depth_2,
                                            inverse_depth_3, inverse_depth_4],
                                    outputs=(reprojections + depth_maps),
                                    name='reprojection_module')
        return reprojection_module


