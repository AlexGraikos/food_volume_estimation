import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Cluster the original 72 food types to reduce learning target complexity
cluster_dict = {
    'patate/pure': 'potatoes', 'pasta_mare_e_monti': 'pasta', 'pizza': 'pizza',
    'budino': 'packaged', 'mandarini': 'fruit', 'pasta_zafferano_e_piselli': 'pasta',
    'arrosto': 'beef', 'yogurt': 'packaged', 'pane': 'bread', 'torta_salata_spinaci_e_ricotta': 'pie',
    'rosbeef': 'beef', 'pizzoccheri': 'pasta', 'arancia': 'fruit', 'carote': 'vegetable',
    'fagiolini': 'vegetable', 'pesce_(filetto)': 'fish', 'spinaci': 'vegetable', 'torta_cioccolato_e_pere': 'pudding',
    'cotoletta': 'chicken', 'patatine_fritte': 'potatoes', 'scaloppine': 'fish',
    'insalata_mista': 'salad', 'insalata_2_(uova mais)': 'salad', 'pasta_sugo': 'pasta',
    'minestra': 'soup', 'mele': 'fruit', 'pasta_bianco': 'pasta', 'riso_bianco': 'rice',
    'pere': 'fruit', 'riso_sugo': 'rice', 'pasta_tonno_e_piselli': 'pasta', 'medaglioni_di_carne': 'beef',
    'pasta_ricotta_e_salsiccia': 'pasta', 'piselli': 'vegetable', 'merluzzo_alle_olive': 'fish',
    'finocchi_in_umido': 'pasta', 'torta_ananas': 'pie', 'passato_alla_piemontese': 'soup',
    'pasta_sugo_vegetariano': 'pasta', 'pasta_tonno': 'pasta', 'cibo_bianco_non_identificato': 'unknown',
    'guazzetto_di_calamari': 'fish', 'stinco_di_maiale': 'pork', 'strudel': 'pudding',
    'zucchine_impanate': 'vegetable', 'zucchine_umido': 'vegetable', 'roastbeef': 'beef',
    'crema_zucca_e_fagioli': 'soup', 'lasagna_alla_bolognese': 'pasta', 'finocchi_gratinati': 'vegetable',
    'pasta_pancetta_e_zucchine': 'pasta', 'rucola': 'salad', 'orecchiette_(ragu)': 'pasta',
    'arrosto_di_vitello': 'beef', 'pasta_e_ceci': 'pasta', 'torta_crema': 'pudding',
    'torta_salata_(alla_valdostana)': 'pie', 'pasta_cozze_e_vongole': 'pasta',
    'banane': 'fruit', 'pasta_pesto_besciamella_e_cornetti': 'pasta', 'pasta_e_fagioli': 'pasta',
    'torta_salata_rustica_(zucchine)': 'pie', 'bruscitt': 'beef', 'focaccia_bianca': 'pie',
    'pesce_2_(filetto)': 'fish', 'torta_crema_2': 'pudding', 'pasta_sugo_pesce': 'pasta',
    'polpette_di_carne': 'beef', 'salmone_(da_menu_sembra_spada_in_realta)': 'fish',
    'cavolfiore': 'vegetable', 'torta_salata_3': 'pie', 'minestra_lombarda': 'soup',
    'patate/pure_prosciutto': 'potatoes'
}
clusters = list(set(val for val in cluster_dict.values())) 
clusters.sort() # To keep the same order across all executions

# Options for training with different cluster arrangements
SINGLE_CLUSTER = True # Overwrite all clusters as "food"
ORIGINAL_LABELS = False # Use original labels
assert not SINGLE_CLUSTER or not ORIGINAL_LABELS, (
    'Cannot apply both SINGLE_CLUSTER and ORIGINAL_LABELS.')

if SINGLE_CLUSTER:
    for key in cluster_dict.keys():
        # If cluster is "packaged" then label as "BG" to ignore
        if cluster_dict[key] == 'packaged':
            cluster_dict[key] = 'BG'
        else:
            cluster_dict[key] = 'food'
    clusters = ['food']

if ORIGINAL_LABELS:
    for key in cluster_dict.keys():
        cluster_dict[key] = key
    clusters = cluster_dict.keys()

# Map each cluster to an integer label
class_id_map = {c: i+1 for i,c in enumerate(clusters)}


# Mask RCNN Configuration
ROOT_DIR = os.path.abspath('../../')
from food_volume_estimation.food_segmentation.mrcnn.config import Config
from food_volume_estimation.food_segmentation.mrcnn import (
    model as modellib, 
    utils)
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, '/models/mask_rcnn_coco.h5')
DEFAULT_LOGS_DIR = 'logs/'


class FoodConfig(Config):
    """Configuration for training on the UNIMIB2016 food dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = 'UNIMIB2016-food'

    # Adjust to appropriate GPU specifications
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(clusters)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 925 // IMAGES_PER_GPU

    # Validation steps per epoch
    VALIDATION_STEPS = 102 // IMAGES_PER_GPU

    # Less ROIs to keep 1:3 ratio
    TRAIN_ROIS_PER_IMAGE = 128


class FoodDataset(utils.Dataset):
    def load_food(self, dataset_dir, subset, annotations_file):
        """Load a subset of the UNIMIB2016 food dataset.

        Inputs:
            dataset_dir: Root directory of the dataset.
            subset: Subset to load, "train" or "val"
        """
        # Add each class with its corresponding id
        for i,c in enumerate(clusters):
            self.add_class('UNIMIB2016-food', i+1, c)
        
        # Load annotations file
        with open(annotations_file, 'r') as json_file:
            annotations = json.load(json_file)

        # Directory to load dataset images from
        assert subset in ['train', 'val']
        dataset_dir = os.path.join(dataset_dir, subset)

        # For each image in the dataset add its path and annotations
        not_found = 0
        for ann in annotations:
            image_path = os.path.join(dataset_dir, ann['filename'] + '.jpg')
            # Skip missing images
            try:
                image = skimage.io.imread(image_path)
            except FileNotFoundError:
                # Uncomment to log all images not found
                # print('[!] Image', image_path, 'was not found.')
                not_found += 1
                continue
            height, width = image.shape[:2]
            
            # Unpack polygon vertex coordinates in a list
            polygons = []
            if isinstance(ann['objects'], list):
                for item in ann['objects']:
                    # Ignore item types with cluster "BG"
                    if cluster_dict[item['type']] != 'BG':
                        polygons.append({'type': cluster_dict[item['type']],
                                         'vertices_x': item['polygon_x'],
                                         'vertices_y': item['polygon_y']})
            else:
                # Ignore item types with cluster "BG"
                if cluster_dict[ann['objects']['type']] != 'BG':
                    polygons.append(
                        {'type': cluster_dict[ann['objects']['type']],
                        'vertices_x': ann['objects']['polygon_x'],
                        'vertices_y': ann['objects']['polygon_y']})

            self.add_image('UNIMIB2016-food', image_id=ann['filename'], 
                           path=image_path, width=width, height=height, 
                           polygons=polygons)

        # Print a warning if some images mentioned in the annotations file
        # were not found in the dataset
        if not_found > 0:
            print('[!]', not_found, 'images were not found in', 
                  '"' + dataset_dir + '"', 'for subset', '"' + subset + '".')

    def load_mask(self, image_id):
        """Generate instance masks for an image.

        Inputs:
            image_id: Image handle to load masks for.
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a food dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info['source'] != 'UNIMIB2016-food':
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info['height'], info['width'], 
                         len(info['polygons'])], dtype=np.uint8)
        class_ids = np.zeros(len(info['polygons']), dtype=np.int32) 
        for i,p in enumerate(info['polygons']):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['vertices_y'], p['vertices_x'])
            mask[rr, cc, i] = 1
            # Get the class id and assign to array
            class_ids[i] = class_id_map[p['type']]

        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image.

        Inputs:
            image_id: Image handle to return info for.
        """
        info = self.image_info[image_id]
        if info['source'] == 'UNIMIB2016-food':
            return info['path']
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, dataset_dir, annotations_file, epochs):
    from imgaug import augmenters as iaa

    """Train the mask rcnn model.

    Inputs:
        model: Model to train.
        dataset_dir: Root directory of dataset.
        epochs: Epochs to train for. If given two values, the network
                heads are first trained for epochs[0] before training
                the full model to epochs[1].
    """
    # Training dataset
    dataset_train = FoodDataset()
    dataset_train.load_food(dataset_dir, 'train', annotations_file)
    dataset_train.prepare()
    print('[*] Training dataset:')
    print(' ', 'Image Count: {}'.format(len(dataset_train.image_ids)))
    print(' ', 'Class Count: {}'.format(dataset_train.num_classes))
    print(' ', 'Classes:', dataset_train.class_names)

    #Validation dataset
    dataset_val = FoodDataset()
    dataset_val.load_food(dataset_dir, 'val', annotations_file)
    dataset_val.prepare()
    print('[*] Validation dataset:')
    print(' ', 'Image Count: {}'.format(len(dataset_val.image_ids)))
    print(' ', 'Class Count: {}'.format(dataset_val.num_classes))
    print(' ', 'Classes:', dataset_val.class_names)

    # Input augmentations
    augmentation = iaa.SomeOf((0, None), [
        iaa.Fliplr(0.5), # Left-right flip with probability 0.5
        iaa.Flipud(0.5), # Up-down flip with probability 0.5
        iaa.Add((-40, 40)), # Add delta value to brightness
        iaa.LinearContrast((0.8, 1.2)), # Transform contrast
        iaa.AddToSaturation((-40, 40)), # Add delta value to saturation
        iaa.AddToHue((-20, 20)) # Add delta value to hue
    ])

    # Train network heads first if two epoch values are given
    if len(epochs) > 1:
        print('[*] Training network heads.')
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    augmentation=augmentation,
                    epochs=epochs[0],
                    layers='heads')
    else:
        epochs.append(epochs[0])

    print('[*] Training network.')
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                augmentation=augmentation,
                epochs=epochs[1],
                layers='all')

def infer(model, image_paths):
    """Infer the model output on a single image.

    Inputs:
        model: Model to use for inference.
        image_paths: List of paths to images to detect food in.
    """
    from food_volume_estimation.food_segmentation.mrcnn import visualize
    from food_volume_estimation.food_segmentation.mrcnn.visualize import display_images
    from food_volume_estimation.food_segmentation.mrcnn.model import log

    for path in image_paths:
        class_names = ['bg'] + clusters
        image = skimage.io.imread(path)
        results = model.detect([image], verbose=0)
        r = results[0]
        visualize.display_instances(image, r['rois'],
                                    r['masks'], r['class_ids'],
                                    class_names, r['scores'])

def eval_on_set(model, dataset_dir, subset, annotations_file, n_evals=3):
    """Show evaluation examples of model on given subset of dataset.

    Inputs:
        model: Model to evaluate.
        dataset_dir: Root directory of dataset.
        subset: Dataset subset to draw samples from.
        n_evals: Number of examples to evaluate.
    """
    from food_volume_estimation.food_segmentation.mrcnn import visualize
    from food_volume_estimation.food_segmentation.mrcnn.visualize import display_images
    from food_volume_estimation.food_segmentation.mrcnn.model import log

    # Evaluation set
    dataset = FoodDataset()
    dataset.load_food(dataset_dir, subset, annotations_file)
    dataset.prepare()
    print('[*] Evaluation dataset:')
    print(' ', 'Image Count: {}'.format(len(dataset.image_ids)))
    print(' ', 'Class Count: {}'.format(dataset.num_classes))
    print(' ', 'Classes:', dataset.class_names)

    for i in range(n_evals): 
        image_id = np.random.choice(dataset.image_ids)
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        # Compute Bounding box
        bbox = utils.extract_bboxes(mask)

        # Display image and additional stats
        print('image_id ', image_id, dataset.image_reference(image_id))
        log('image', image)
        log('mask', mask)
        log('class_ids', class_ids)
        log('bbox', bbox)
        # Display image and instances
        visualize.display_instances(image, bbox, mask, class_ids,
                                    dataset.class_names)
        original_image = image

        results = model.detect([original_image], verbose=0)
        r = results[0]
        visualize.display_instances(original_image, r['rois'],
                                    r['masks'], r['class_ids'],  
                                    dataset.class_names, r['scores'])


if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect food.')
    parser.add_argument('command',
                        metavar='<command>',
                        help='Command "train", "infer" or "eval".')
    parser.add_argument('--dataset', required=False,
                        metavar='/path/to/food/dataset/',
                        help='Directory of the food dataset.')
    parser.add_argument('--annotations', required=False,
                        metavar='/path/to/dataset/annotations.json',
                        help='Dataset annotations file.')
    parser.add_argument('--weights', required=True,
                        metavar='/path/to/weights.h5',
                        help='Path to weights .h5 file or "coco".')
    parser.add_argument('--epochs', required=False, nargs='+',
                        type=int, default=10,
                        metavar='<epochs>',
                        help='Number of epochs to train for. '
                             'When given two values e1,e2 the network heads '
                             'are first trained for e1 epochs and then '
                             'the training continues to e2 epochs.')
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar='/path/to/logs/',
                        help='Logs and checkpoints directory.')
    parser.add_argument('--images', required=False, nargs='+',
                        metavar='/path/1 /path/2 ...',
                        help='Path to one or more images to detect food in.')
    parser.add_argument('--n_evals', required=False, type=int,
                        metavar='<number of evaluations>',
                        help='Number of images to evaluate results on.')
    args = parser.parse_args()

    # Validate arguments
    if args.command == 'train':
        assert args.dataset, 'Argument --dataset is required for training'
        assert args.annotations, ('Argument --annotations is required for '
                                  'training')
    elif args.command == 'infer':
        assert args.images, 'Provide --image to detect food in'
    elif args.command == 'eval':
        assert args.dataset, 'Argument --dataset is required for evaluating'
        assert args.annotations, ('Argument --annotations is required for '
                                  'evaluating')

    print('[*] Weights:', '"' + args.weights + '".')
    if args.command == 'infer':
        print('[*] Images:', args.images)
    else:
        print('[*] Dataset:', '"' + args.dataset + '".')
    print('[*] Logs:', '"' + args.logs + '".')

    # Configurations
    if args.command == 'train':
        config = FoodConfig()
    else:
        class InferenceConfig(FoodConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == 'train':
        model = modellib.MaskRCNN(mode='training', config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode='inference', config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == 'coco':
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == 'last':
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == 'imagenet':
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print('[*] Loading weights ', weights_path)
    if args.weights.lower() == 'coco':
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            'mrcnn_class_logits', 'mrcnn_bbox_fc',
            'mrcnn_bbox', 'mrcnn_mask'])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train, infer or evaluate
    if args.command == 'train':
        train(model, args.dataset, args.annotations, args.epochs)
    elif args.command == 'infer':
        infer(model, args.images)
    elif args.command == 'eval':
        eval_on_set(model, args.dataset, 'val', args.annotations, args.n_evals)
    else:
        print('[!] "{}" is not recognized. '
              'Use "train", "infer" or "eval".'.format(args.command))

