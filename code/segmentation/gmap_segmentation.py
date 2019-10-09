from __future__ import print_function
from __future__ import division


import argparse
import keras
from keras.regularizers import l2
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.preprocessing import image
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import cv2



def load_image(img_path, dimentions, rescale=1. / 255):
    img = load_img(img_path, target_size=dimentions)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x *= rescale # rescale the same as when trained

    return x
def get_classes(file_path):
    with open(file_path) as f:
        classes = f.read().splitlines()

    return classes
def create_model(num_classes, dropout, shape):


    base_model = VGG16(include_top=False, input_tensor=Input(shape=shape))
    model = base_model.output

    model = ZeroPadding2D((1,1))(model)
    model = Conv2D(1024, (3, 3), activation='relu')(model)
    model = MaxPooling2D((4,4), strides=(2,2))(model)
    model = GlobalAveragePooling2D()(model)
#     model = Dense(4096, activation='relu')(model)
    model = Dropout(dropout)(model)
    predictions = Dense(num_classes, activation='softmax')(model)
    model_final = Model(inputs=base_model.input, outputs=predictions)

    return model_final
def load_model(model_final, weights_path, shape, dropout):
   model_final = create_model(101, dropout, shape)
   model_final.load_weights(weights_path)

   return model_final



def get_classmap_numpy(model, img, label):

    _ , width, height, _ = img.shape

    # Get the 1024 input weights to softmax
    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = model.layers[-5]
    get_output = K.function([model.layers[0].input, keras.backend.learning_phase()], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img, 0])
    conv_outputs = conv_outputs[0,:,:,:]

    #create the class activation map
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
    for i,w in enumerate(class_weights[:,label]):
        cam += w*conv_outputs[:,:,i]


    cam = cv2.resize(cam, (height, width))
#     debug = cam/np.max(cam)

    return cam

def plot_cam3(original_image, cam):

    height, width ,_ = original_image.shape
    cam_norm = cam / np.max(cam)
    cam_resize = cv2.resize(cam_norm, (width, height))
    cam_resize = cam_resize*255
    cam_resize = cam_resize.astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_resize, cv2.COLORMAP_JET)
    heatmap[np.where(cam_resize < 0.2)] = 0
    # cv2.imshow("heatmap", heatmap)
    # cv2.waitKey(100000)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

    superimposed_img = cv2.addWeighted(original_image, 0.7, heatmap, 0.5, 0)
    # cv2.imshow("cam3", superimposed_img)
    # cv2.waitKey(100000)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

    return heatmap, superimposed_img


def get_segment(original_image, heatmap):

    # color boundaries
    boundary_red = [([0, 0, 1], [0, 80, 255])]
    boundary_yellow = [([0, 80, 130 ], [140, 255, 255])]
    boundary_lightblue = [([130, 130, 0], [255, 255, 130])]
    boundary_blue = [([130, 0, 0], [255, 130, 0])]

    lower, upper = boundary_red[0]
    lower_red = np.array(lower, dtype = "uint8")
    upper_red = np.array(upper, dtype = "uint8")
    lower, upper = boundary_lightblue[0]
    lower_lightblue = np.array(lower, dtype = "uint8")
    upper_lightblue = np.array(upper, dtype = "uint8")
    lower, upper = boundary_blue[0]
    lower_blue = np.array(lower, dtype = "uint8")
    upper_blue = np.array(upper, dtype = "uint8")
    lower, upper = boundary_yellow[0]
    lower_yellow = np.array(lower, dtype = "uint8")
    upper_yellow = np.array(upper, dtype = "uint8")


    # detection of colors in image
    mask_red = cv2.inRange(heatmap, lower_red, upper_red)
    mask_yellow = cv2.inRange(heatmap, lower_yellow, upper_yellow)
    mask_lightblue = cv2.inRange(heatmap, lower_lightblue, upper_lightblue)
    mask_blue = cv2.inRange(heatmap, lower_blue, upper_blue)

    # grabCut initialization
    mask = np.zeros(original_image.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    mask[mask_red == 255] = 1
    mask[mask_yellow == 255] = 3
    mask[mask_lightblue == 255] = 2
    mask[mask_blue == 255] = 0

    # reduce noise in the edge of the image (800x600 pixels)
    # columns pass for noise elimination
    height, width = mask.shape
    row_filter = int(0.15 * height)
    column_filter = int(0.15 * width)
    mask[:row_filter, :] = 0
    mask[-row_filter:, :] = 0
    mask[:, :column_filter] = 0
    mask[:, -column_filter:] = 0


    # segmentation of the food and creation of the mask to use for the histogram calculation
    mask_hist = np.zeros(original_image.shape[:2],np.uint8)
    mask, bgdModel, fgdModel = cv2.grabCut(original_image,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    mask_hist[mask == 1] = 255
    segment = original_image*mask[:,:,np.newaxis]

    return segment, mask_hist
def get_segment_loose(original_image, heatmap):

    # color boundaries
    boundary_red = [([0, 0, 1], [0, 80, 255])]
    boundary_yellow = [([0, 80, 130 ], [140, 255, 255])]
    boundary_lightblue = [([130, 130, 0], [255, 255, 130])]
    boundary_blue = [([130, 0, 0], [255, 130, 0])]

    lower, upper = boundary_red[0]
    lower_red = np.array(lower, dtype = "uint8")
    upper_red = np.array(upper, dtype = "uint8")
    lower, upper = boundary_lightblue[0]
    lower_lightblue = np.array(lower, dtype = "uint8")
    upper_lightblue = np.array(upper, dtype = "uint8")
    lower, upper = boundary_blue[0]
    lower_blue = np.array(lower, dtype = "uint8")
    upper_blue = np.array(upper, dtype = "uint8")
    lower, upper = boundary_yellow[0]
    lower_yellow = np.array(lower, dtype = "uint8")
    upper_yellow = np.array(upper, dtype = "uint8")

    # detection of colors in image
    mask_red = cv2.inRange(heatmap, lower_red, upper_red)
    mask_yellow = cv2.inRange(heatmap, lower_yellow, upper_yellow)
    mask_lightblue = cv2.inRange(heatmap, lower_lightblue, upper_lightblue)
    mask_blue = cv2.inRange(heatmap, lower_blue, upper_blue)

    # grabCut initialization
    mask = np.zeros(original_image.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    mask[mask_red == 255] = 1
    mask[mask_yellow == 255] = 3
    mask[mask_lightblue == 255] = 3
    mask[mask_blue == 255] = 0

    # reduce noise in the edge of the image  (800x600 pixels)
    mask[1:100] = 0
    mask[470:600] = 0
    for i in range(600):
        mask[i][1:100] = 0
        mask[i][700:800] = 0



    # segmentation of the food and creation of the mask to use for the histogram calculation
    mask_hist = np.zeros(original_image.shape[:2],np.uint8)
    mask, bgdModel, fgdModel = cv2.grabCut(original_image,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    mask_hist[mask == 1] = 255
    segment = original_image*mask[:,:,np.newaxis]

    return segment, mask_hist
def get_segment_tight(original_image, heatmap):

    # color boundaries
    boundary_red = [([0, 0, 1], [0, 80, 255])]
    boundary_yellow = [([0, 80, 130 ], [140, 255, 255])]
    boundary_lightblue = [([130, 130, 0], [255, 255, 130])]
    boundary_blue = [([130, 0, 0], [255, 130, 0])]

    lower, upper = boundary_red[0]
    lower_red = np.array(lower, dtype = "uint8")
    upper_red = np.array(upper, dtype = "uint8")
    lower, upper = boundary_lightblue[0]
    lower_lightblue = np.array(lower, dtype = "uint8")
    upper_lightblue = np.array(upper, dtype = "uint8")
    lower, upper = boundary_blue[0]
    lower_blue = np.array(lower, dtype = "uint8")
    upper_blue = np.array(upper, dtype = "uint8")
    lower, upper = boundary_yellow[0]
    lower_yellow = np.array(lower, dtype = "uint8")
    upper_yellow = np.array(upper, dtype = "uint8")

    # detection of colors in image
    mask_red = cv2.inRange(heatmap, lower_red, upper_red)
    mask_yellow = cv2.inRange(heatmap, lower_yellow, upper_yellow)
    mask_lightblue = cv2.inRange(heatmap, lower_lightblue, upper_lightblue)
    mask_blue = cv2.inRange(heatmap, lower_blue, upper_blue)


    # grabCut initialization
    mask = np.zeros(original_image.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    mask[mask_red == 255] = 1
    mask[mask_yellow == 255] = 2
    mask[mask_lightblue == 255] = 0
    mask[mask_blue == 255] = 0

    # reduce noise in the edge of the image (800x600 pixels)
    # mask[1:130] = 0
    # mask[470:600] = 0
    # for i in range(600):
    #     mask[i][1:100] = 0
    #     mask[i][700:800] = 0

    # segmentation of the food and creation of the mask to use for the histogram calculation
    mask_hist = np.zeros(original_image.shape[:2],np.uint8)
    mask, bgdModel, fgdModel = cv2.grabCut(original_image,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    mask_hist[mask == 1] = 255
    segment = original_image*mask[:,:,np.newaxis]

    return segment, mask_hist


def get_food_segmentation(image_path, weights_path, segmentation_method='normal'):
    # initialization of the model
    shape = (224, 224, 3)
    dropout=0.5
    trained_model = create_model(101, dropout, shape)
    trained_model.load_weights(weights_path)

    # loading of the image
    original_image = cv2.imread(image_path)

    # load the image in keras form and predict label
    image = load_image(image_path , shape[:2])
    preds = trained_model.predict(image)

    # get network activation for image and class
    cam = get_classmap_numpy(trained_model, image, np.argmax(preds))

    # get heatmap and heatmap on top of the original image
    heatmap, superimposed_img = plot_cam3(original_image, cam)

    # get a normal segment of the food along with its binary mask + visualization
    if segmentation_method == 'normal':
        segmented_food, mask = get_segment(original_image, heatmap)
    elif segmentation_method == 'tight':
        segmented_food, mask = get_segment_tight(original_image, heatmap)
    elif segmentation_method == 'loose':
        segmented_food, mask = get_segment_loose(original_image, heatmap)
    else:
        print('[!] Invalid segmentation method')

    return mask


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description='GMAP food segmentation')
    parser.add_argument('--image', type=str,
                        help='the file path of the input image')
    # parser.add_argument('--output', type=str, default="segmented_food.jpg",
    #                     help='the file path of the output image')
    parser.add_argument('--type', type=int, default= 1,
                        help='type of segmentation to use. 1 for normal, 2 for tight, 3 for loose')
    args = parser.parse_args()

    # initialization of the model
    shape = (224, 224, 3)
    dropout=0.5
    trained_model = create_model(101, dropout, shape)
    trained_model.load_weights('weights.epoch-21-val_loss-0.85.hdf5')

    # loading of the image
    image_path = args.image
    original_image = cv2.imread(image_path)
    # height, width ,_ = original_image.shape
    classes = get_classes('../meta/classes.txt')

    # load the image in keras form and predict label
    image = load_image(image_path , shape[:2])
    preds = trained_model.predict(image)
    # print("The index is", np.argmax(preds))
    # print("The class is: ", classes[np.argmax(preds)])
    # print("The possibility is", preds[0][np.argmax(preds)])


    # get network activation for image and class
    cam = get_classmap_numpy(trained_model, image, np.argmax(preds))

    # get heatmap and heatmap on top of the original image
    heatmap, superimposed_img = plot_cam3(original_image, cam)

    # visualization of CAM and image with location
    # cv2.imshow("Heatmap", heatmap)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # cv2.imshow("Localization", superimposed_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    if args.type == 1:
        # get a normal segment of the food along with its binary mask + visualization
        segmented_food, mask = get_segment(original_image, heatmap)
        cv2.imshow("Segmented Food normal", np.hstack([superimposed_img, segmented_food]))
        cv2.waitKey()
        cv2.destroyAllWindows()
    elif args.type == 2:
        # get a tight segment of the food along with its binary mask + visualization
        segmented_food_tight, mask_tight = get_segment_tight(original_image, heatmap)
        cv2.imshow("Segmented Food tight", np.hstack([superimposed_img, segmented_food_tight]))
        cv2.waitKey()
        cv2.destroyAllWindows()
    elif args.type == 3:
        # get a loose segment of the food along with its binary mask + + visualization
        segmented_food_loose, mask_loose = get_segment_loose(original_image, heatmap)
        cv2.imshow("Segmented Food loose", np.hstack([superimposed_img, segmented_food_loose]))
        cv2.waitKey()
        cv2.destroyAllWindows()





    # every type of segmentation and side by side comparison


    # # get a normal segment of the food along with its binary mask + visualization
    # segmented_food, mask = get_segment(original_image, heatmap)
    # cv2.imshow("Segmented Food normal", segmented_food)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # # get a tight segment of the food along with its binary mask + visualization
    # segmented_food_tight, mask_tight = get_segment_tight(original_image, heatmap)
    # cv2.imshow("Segmented Food tight", segmented_food_tight)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # # get a loose segment of the food along with its binary mask + + visualization
    # segmented_food_loose, mask_loose = get_segment_loose(original_image, heatmap)
    # cv2.imshow("Segmented Food loose", segmented_food_loose)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # # visualize the original image and every kind of segmentation side by side
    # cv2.imshow("Segmented Food", np.hstack([superimposed_img, segmented_food, segmented_food_tight, segmented_food_loose]))
    # cv2.waitKey()
    # cv2.destroyAllWindows()




    # # write jpgs of the above visualizations
    # cv2.imwrite('CAM.jpg', heatmap)
    # cv2.imwrite('food_lcalization.jpg', superimposed_img)
    # cv2.imwrite('segmented_food.jpg', segmented_food)
    # cv2.imwrite('segmented_food_tight.jpg', segmented_food_tight)
    # cv2.imwrite('segmented_food_loose.jpg', segmented_food_loose)
