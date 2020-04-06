import sys
import os
from food_volume_estimation.food_segmentation.mrcnn.config import Config
from food_volume_estimation.food_segmentation.mrcnn import (
    model as modellib,
    utils)

# Using the single-cluster model
clusters = ['food']

class FoodSegConfig(Config):
    """Configuration for inferring segmentation masks using the 
    model trained on the UNIMIB2016 food dataset. 
    """
    # Give the configuration a recognizable name
    NAME = 'UNIMIB2016-food'

    # Adjust to appropriate GPU specifications
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(clusters)

class FoodSegmentator():
    """Food segmentator object using the Mask RCNN model."""
    def __init__(self, weights_path):
        """Initialize the segmentation model.

        Inputs:
            weights_path: Path to model weights file (.h5).
        """
        # Create model and load weights
        config = FoodSegConfig()
        self.model = modellib.MaskRCNN(mode='inference', config=config,
                                  model_dir='')
        print('[*] Loading segmentation model weights', weights_path)
        self.model.load_weights(weights_path, by_name=True)

    def infer_masks(self, input_image):
        """Infer the segmentation masks in the input image.

        Inputs:
            input_image: Path to image or image array to detect food in.
        Returns:
            masks: [m,n,k] array containing each of the K masks detected.
        """
        import cv2

        # Load image and infer masks
        if isinstance(input_image, str):
            image = cv2.imread(input_image, cv2.IMREAD_COLOR)
        else:
            image = input_image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model.detect([image], verbose=0)
        r = results[0]
        masks = r['masks'].astype('float32') # openCV can't resize int images

        return masks

    def infer_and_plot(self, image_paths):
        """Infer the model output on a single image and plot the results.

        Inputs:
            image_paths: List of paths to images to detect food in.
        """
        import cv2
        from food_volume_estimation.food_segmentation.mrcnn import visualize
        from food_volume_estimation.food_segmentation.mrcnn.visualize import display_images

        for path in image_paths:
            class_names = ['bg'] + clusters
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.model.detect([image], verbose=0)
            r = results[0]
            visualize.display_instances(image, r['rois'],
                                        r['masks'], r['class_ids'],
                                        class_names, r['scores'])

if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Infer food segmentation masks using the '
                    'Mask R-CNN model.')
    parser.add_argument('--weights', required=True,
                        metavar='/path/to/weights.h5',
                        help='Path to weights .h5 file.')
    parser.add_argument('--images', required=False, nargs='+',
                        metavar='/path/1 /path/2 ...',
                        help='Path to one or more images to detect food in.')
    args = parser.parse_args()
    
    # Create segmentator object
    seg_model = FoodSegmentator(args.weights) 
    # Infer segmentation masks and plot results
    seg_model.infer_and_plot(args.images)
