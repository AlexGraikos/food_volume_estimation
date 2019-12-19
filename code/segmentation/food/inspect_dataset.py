import os
import sys
import argparse
import json
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath('../')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.model import log
# Import food dataset utils
from food_instance_segmentation import FoodConfig, FoodDataset, cluster_dict


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Inspect food dataset.')
    parser.add_argument('--dataset', required=True,
                        metavar='/path/to/food/dataset/',
                        help='Root directory of the food dataset.')
    parser.add_argument('--subset', required=True,
                        metavar='<subset_dir>',
                        help='Subset directory, "train" or "val".')
    parser.add_argument('--n_reps', required=False, type=int,
                        metavar='<number of evaluations>',
                        default=3,
                        help='Number of images to inspect.')
    args = parser.parse_args()

    # Load configuration and dataset 
    assert args.subset in ['train', 'val'], (
           'Subset must be either "train" or "val".')
    config = FoodConfig()
    dataset = FoodDataset()
    dataset.load_food(args.dataset, args.subset)
    dataset.prepare()
    
    # Show general dataset information 
    print('[*] Image Count: {}'.format(len(dataset.image_ids)))
    print('[*] Class Count: {}'.format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print('{:3}. {:50}'.format(i, info['name']))

    # Load annotations
    annotations_file = os.path.join(args.dataset, 'annotations.json')
    with open(annotations_file, 'r') as json_file:
        annotations = json.load(json_file)

    # Count number of instances per original and clustered label
    instance_count_dict = dict()
    cluster_count_dict = dict()
    for ann in annotations:
        if isinstance(ann['objects'], list):
            for item in ann['objects']:
                # Instance counting
                if item['type'] in instance_count_dict.keys():
                    instance_count_dict[item['type']] += 1
                else:
                    instance_count_dict[item['type']] = 1
                # Cluster counting 
                if cluster_dict[item['type']] in cluster_count_dict.keys():
                    cluster_count_dict[cluster_dict[item['type']]] += 1
                else:
                    cluster_count_dict[cluster_dict[item['type']]] = 1
        else:
            # Instance counting
            if ann['objects']['type'] in instance_count_dict.keys():
                instance_count_dict[ann['objects']['type']] += 1
            else:
                instance_count_dict[ann['objects']['type']] = 1
            # Cluster counting 
            if cluster_dict[ann['objects']['type']] in cluster_count_dict.keys():
                cluster_count_dict[cluster_dict[ann['objects']['type']]] += 1
            else:
                cluster_count_dict[cluster_dict[ann['objects']['type']]] = 1
    print('[*] Dataset instance label count:\n', instance_count_dict, sep='')
    print('[*] Dataset clustered instance label count:\n', 
          cluster_count_dict, sep='')

    # Plot random dataset examples
    # (Image with overlayed class bounding boxes and masks)
    for i in range(args.n_reps):
        # Load random image and mask
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

