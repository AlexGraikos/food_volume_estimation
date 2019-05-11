import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D, GlobalAveragePooling2D, \
    LeakyReLU, Dense, Flatten, Input, Concatenate, Lambda, BatchNormalization
from classification_models.resnet import ResNet18

import custom_modules

class Networks:
    def __init__(self, img_shape):
        self.img_shape = img_shape


    def create_generator(self):
        """
        Creates and returns the generator network
        """
        # Generator modules
        depth_net = self.__create_depth_net()
        pose_net = self.__create_pose_net()
        reprojection_module = self.__create_reprojection_module()

        # Synthesize generator model
        curr_frame = Input(shape=self.img_shape)
        prev_frame = Input(shape=self.img_shape)
        next_frame = Input(shape=self.img_shape)
        # Depth
        inverseDepths = depth_net(curr_frame)
        # Poses
        pose_prev = pose_net([prev_frame, curr_frame])
        pose_next = pose_net([next_frame, curr_frame])
        # Reprojections
        reprojections = reprojection_module([prev_frame, next_frame, pose_prev, pose_next] + inverseDepths)
        # Concatenate reprojections per-scale for computing per scale min loss
        perScaleReprojections = \
            [Concatenate(name='scale1_reprojections')([reprojections[0], reprojections[1]]),
             Concatenate(name='scale2_reprojections')([reprojections[2], reprojections[3]]),
             Concatenate(name='scale3_reprojections')([reprojections[4], reprojections[5]]),
             Concatenate(name='scale4_reprojections')([reprojections[6], reprojections[7]])]

        generator = Model(inputs=[curr_frame, prev_frame, next_frame],
            outputs=(reprojections + inverseDepths + perScaleReprojections), name='generator')
        return generator


    def __create_depth_net(self):
        """
        Creates and returns the depth predicting network model
            Outputs:
                depth_net: Depth predicting network model
        """
        # ResNet18 encoder
        depth_encoder = ResNet18(input_shape=self.img_shape, weights='imagenet', include_top=False)
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
            dec = custom_modules.ReflectionPadding2D(padding=(1,1))(dec)
            dec = Conv2D(filters=filters, kernel_size=3, activation='elu')(dec)
            return dec

        def inverse_depth_layer(prev_layer):
            inverseDepth = custom_modules.ReflectionPadding2D(padding=(1,1))(prev_layer)
            inverseDepth = Conv2D(filters=1, kernel_size=3, activation='sigmoid')(inverseDepth)
            return inverseDepth
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
        inverseDepth4 = inverse_depth_layer(iconv4)
        inverseDepth3 = inverse_depth_layer(iconv3)
        inverseDepth2 = inverse_depth_layer(iconv2)
        inverseDepth1 = inverse_depth_layer(iconv1)
        # Model
        depth_net = Model(inputs=depth_encoder.input,
            outputs=[inverseDepth1, inverseDepth2, inverseDepth3, inverseDepth4], name='depth_net')
        return depth_net


    def __create_pose_net(self):
        """
        Creates and returns the pose estimation network
            Outputs:
                pose_net: Pose estimation network model
        """
        # Pose encoder
        pose_encoder = ResNet18(input_shape=(self.img_shape[0], self.img_shape[1], 6), weights=None, include_top=False)
        # Inputs
        source_frame = Input(shape=self.img_shape)
        target_frame = Input(shape=self.img_shape)
        concatenated_frames = Concatenate()([source_frame, target_frame])
        # Pose net
        #input_pre = Conv2D(filters=3, kernel_size=3, padding='same')(concatenated_frames)
        pose_features = pose_encoder(concatenated_frames)
        pconv0 = Conv2D(filters=256, kernel_size=1, padding='same', activation='relu')(pose_features)
        pconv1 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(pconv0)
        pconv2 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(pconv1)
        pconv3 = Conv2D(filters=6,   kernel_size=1, padding='same', activation='linear')(pconv2)
        pose = GlobalAveragePooling2D()(pconv3)
        pose_net = Model(inputs=[source_frame, target_frame], outputs=pose, name='pose_net')
        #pose_net.summary()
        return pose_net


    def __create_reprojection_module(self):
        """
        Creates and returns the reprojection module model
            Outputs:
                reprojection_module: Reprojection module model
        """
        prev_frame = Input(shape=self.img_shape)
        next_frame = Input(shape=self.img_shape)
        posePrev = Input(shape=(6,))
        poseNext = Input(shape=(6,))
        inverseDepth1 = Input(shape=(self.img_shape[0],    self.img_shape[1],    1))
        inverseDepth2 = Input(shape=(self.img_shape[0]//2, self.img_shape[1]//2, 1))
        inverseDepth3 = Input(shape=(self.img_shape[0]//4, self.img_shape[1]//4, 1))
        inverseDepth4 = Input(shape=(self.img_shape[0]//8, self.img_shape[1]//8, 1))

        x_scaling = self.img_shape[1]/1920
        y_scaling = self.img_shape[0]/1080
        intrinsicsMatrix = np.array([[1000*x_scaling, 0, 950*x_scaling],
            [0, 1000*y_scaling, 540*y_scaling], [0, 0, 1]])
        #intrinsicsMatrix = np.array([[1, 0, self.img_shape[0]//2],
        #    [0, 1, self.img_shape[1]//2], [0, 0, 1]])

        # Upsample and normalize inverse depth maps
        inverseDepth2Up = UpSampling2D(size=(2,2))(inverseDepth2)
        inverseDepth3Up = UpSampling2D(size=(4,4))(inverseDepth3)
        inverseDepth4Up = UpSampling2D(size=(8,8))(inverseDepth4)
        depthMap1 = Lambda(custom_modules.inverseDepthNormalization)(inverseDepth1)
        depthMap2 = Lambda(custom_modules.inverseDepthNormalization)(inverseDepth2Up)
        depthMap3 = Lambda(custom_modules.inverseDepthNormalization)(inverseDepth3Up)
        depthMap4 = Lambda(custom_modules.inverseDepthNormalization)(inverseDepth4Up)

        # Create reprojections for each depth map scale from highest to lowest
        reprojections = []
        for depthMap in [depthMap1, depthMap2, depthMap3, depthMap4]:
            prevToTarget = custom_modules.ProjectionLayer(intrinsicsMatrix)([prev_frame, depthMap, posePrev])
            nextToTarget = custom_modules.ProjectionLayer(intrinsicsMatrix)([next_frame, depthMap, poseNext])
            reprojections += [prevToTarget, nextToTarget]

        reprojection_module = Model(inputs=[prev_frame, next_frame, posePrev, poseNext,
            inverseDepth1, inverseDepth2, inverseDepth3, inverseDepth4], outputs=reprojections,
            name='reprojection_module')
        return reprojection_module


    def create_discriminator(self):
        """
        Creates and returns the discriminator network
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
        discriminator = Model(inputs=[target_img, reprojected_img1, reprojected_img2,
            reprojected_img3, reprojected_img4], outputs=validation, name='discriminator')
        #discriminator.summary()
        return discriminator


