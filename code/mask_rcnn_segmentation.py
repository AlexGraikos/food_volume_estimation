import sys
import os

# Setup Mask-RCNN library
MASK_RCNN_DIR = os.path.abspath(
    '/home/alex/Projects/food_segmentation/food_instance_segmentation')
sys.path.append(MASK_RCNN_DIR)  # To find Mask RCNN library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

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
    """Food segmentator object using the Mask RCNN model.
    """
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

    def infer_masks(self, image_path):
        """Infer the segmentation masks in the input image.
        Inputs:
            image_path: Path to image to detect food in.
        Returns:
            masks: [m,n,k] array containing each of the K masks detected.
        """
        import skimage.draw

        # Load image and infer masks
        image = skimage.io.imread(image_path)
        results = self.model.detect([image], verbose=0)
        r = results[0]
        masks = r['masks'].astype('float32') # openCV can't resize int images

        return masks

    def infer_and_plot(self, image_paths):
        """Infer the model output on a single image and plot the results.
        Inputs:
            image_paths: List of paths to images to detect food in.
        """
        import skimage.draw
        from mrcnn import visualize
        from mrcnn.visualize import display_images

        for path in image_paths:
            class_names = ['bg'] + clusters
            image = skimage.io.imread(path)
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
