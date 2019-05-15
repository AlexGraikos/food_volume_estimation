import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D, GlobalAveragePooling2D, \
    LeakyReLU, Dense, Flatten, Input, Concatenate, Lambda, BatchNormalization
from classification_models.resnet import ResNet18
from custom_modules import *


class Networks:
    def __init__(self, img_shape):
        self.img_shape = img_shape


    def create_full_model(self):
        """
        Creates the full monocular depth estimation model.
            Outputs:
                full_model: The created model with all available outputs.
        """
        # Create modules
        depth_net = self.__create_depth_net()
        pose_net = self.__create_pose_net()
        reprojection_module = self.__create_reprojection_module()

        # Synthesize model
        curr_frame = Input(shape=self.img_shape)
        prev_frame = Input(shape=self.img_shape)
        next_frame = Input(shape=self.img_shape)
        # Depth
        inverse_depths = depth_net(curr_frame)
        # Poses
        pose_prev = pose_net([prev_frame, curr_frame])
        pose_next = pose_net([next_frame, curr_frame])
        # Reprojections
        reprojections = reprojection_module([prev_frame, next_frame,
                                             pose_prev, pose_next] 
                                            + inverse_depths)
        # Concatenate reprojections per-scale for computing per scale min loss
        per_scale_reprojections = [
            Concatenate(name='scale1_reprojections')([prev_frame,
                                                      next_frame, 
                                                      reprojections[0],
                                                      reprojections[1]]),
            Concatenate(name='scale2_reprojections')([prev_frame,
                                                      next_frame,
                                                      reprojections[2],
                                                      reprojections[3]]),
            Concatenate(name='scale3_reprojections')([prev_frame,
                                                      next_frame,
                                                      reprojections[4],
                                                      reprojections[5]]),
            Concatenate(name='scale4_reprojections')([prev_frame,
                                                      next_frame,
                                                      reprojections[6],
                                                      reprojections[7]])]
        full_model = Model(
            inputs=[curr_frame, prev_frame, next_frame],
            outputs=(reprojections + inverse_depths
                     + per_scale_reprojections),
            name='full_model')
        return full_model


    def __create_depth_net(self):
        """
        Creates the depth predicting network model.
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
            dec = Conv2D(filters=filters, kernel_size=3, activation='elu')(dec)
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
        """
        Creates and returns the pose estimation network.
            Outputs:
                pose_net: Pose estimation network model.
        """
        # Pose encoder
        pose_encoder = ResNet18(input_shape=(self.img_shape[0],
                                             self.img_shape[1], 6),
                                weights=None, include_top=False)

        # Copy pre-trained ResNet18 weights to 6-channel pose encoder
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
        pose_net = Model(
            inputs=[source_frame, target_frame],
            outputs=pose, 
            name='pose_net')
        return pose_net


    def __create_reprojection_module(self):
        """
        Creates and returns the reprojection module model.
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
        # Scale intrinsics matrix to image size
        x_scaling = self.img_shape[1] / 1920
        y_scaling = self.img_shape[0] / 1080
        intrinsics_mat = np.array([[1000*x_scaling, 0, 950*x_scaling],
                                   [0, 1000*y_scaling, 540*y_scaling],
                                   [0, 0, 1]])
        # Upsample and normalize inverse depth maps
        inverse_depth_2_up = UpSampling2D(size=(2,2))(inverse_depth_2)
        inverse_depth_3_up = UpSampling2D(size=(4,4))(inverse_depth_3)
        inverse_depth_4_up = UpSampling2D(size=(8,8))(inverse_depth_4)
        depth_map_1 = InverseDepthNormalization(0.1, 10)(inverse_depth_1)
        depth_map_2 = InverseDepthNormalization(0.1, 10)(inverse_depth_2_up)
        depth_map_3 = InverseDepthNormalization(0.1, 10)(inverse_depth_3_up)
        depth_map_4 = InverseDepthNormalization(0.1, 10)(inverse_depth_4_up)

        # Create reprojections for each depth map scale from highest to lowest
        reprojections = []
        for depth_map in [depth_map_1, depth_map_2, depth_map_3, depth_map_4]:
            prev_to_target = ProjectionLayer(intrinsics_mat)([prev_frame,
                                                              depth_map,
                                                              pose_prev])
            next_to_target = ProjectionLayer(intrinsics_mat)([next_frame,
                                                              depth_map,
                                                              pose_next])
            reprojections += [prev_to_target, next_to_target]

        reprojection_module = Model(inputs=[prev_frame, next_frame,
                                            pose_prev, pose_next,
                                            inverse_depth_1, inverse_depth_2,
                                            inverse_depth_3, inverse_depth_4],
                                    outputs=reprojections,
                                    name='reprojection_module')
        return reprojection_module


