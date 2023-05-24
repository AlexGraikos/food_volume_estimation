from collections import OrderedDict
from pycocotools.coco import COCO
import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
import tqdm
import cv2
import random
# from keras.preprocessing.image import ImageDataGenerator

# heavily based on the tutorial from Viraf Patrawala (March, 2020)
# https://github.com/virafpatrawala/COCO-Semantic-Segmentation/blob/master/COCOdataset_SemanticSegmentation_Demo.ipynb


TRAIN_ANNOTATIONS_PATH = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_training_set_release_2.0/annotations.json"
TRAIN_IMAGE_DIRECTORY = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_training_set_release_2.0/images/"

VAL_ANNOTATIONS_PATH = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_validation_set_2.0/annotations.json"
VAL_IMAGE_DIRECTORY = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_validation_set_2.0/images/"


def fix_uneven_data(input_annotation_file, out_annotation_file: str, image_dir: str):
    assert os.path.isdir(image_dir), f"Fix annotations: Provided path for images is invalid!"
    assert os.path.isfile(input_annotation_file) is not None, f"Fix annotations: Given input filepath is invalid!"
    coco_obj = COCO(input_annotation_file)
    annotations = coco_obj.loadAnns(coco_obj.getAnnIds())

    for n, i in enumerate(tqdm((annotations['images']))):
        img = cv2.imread(image_dir+i["file_name"])
        if img.shape[0] != i['height']:
            annotations['images'][n]['height'] = img.shape[0]
        if img.shape[1] != i['width']:
            annotations['images'][n]['width'] = img.shape[1]
    with open(out_annotation_file, 'w') as f:
        json.dump(annotations, f)

def visualize_datasplit(img_per_cat):
    fig = go.Figure([go.Bar(x=list(img_per_cat.keys()), y=list(img_per_cat.values()))])
    fig.update_layout(
        title="No of Image per class", )
    fig.show()
    fig = go.Figure(data=[go.Pie(labels=list(img_per_cat.keys()), values=list(img_per_cat.values()),
                                 hole=.3, textposition='inside', )], )
    fig.update_layout(
        title="No of Image per class ( In pie )", )
    fig.show()

class CocoDatasetGenerator:

    def __init__(self, annotations_path: str, img_dir: str, filter_categories=None, data_size=(455,455)) -> None:
        """ Initialize CocoDatasetGenerator

        Args:
            annotations_path (str): Absolute path to the coco annotations file
            img_dir (str): Absolute path to the corresponding image directory
            filter_categories (list, optional): List of category names to use. Uses all if None. Defaults to None.
            data_size (tuple, optional): The size of the generator output images and masks. Defaults to (455,455).
        """
        assert os.path.isfile(annotations_path), f"Provided path for train annotations is invalid!"
        self.annotations_path = annotations_path
        self.img_dir = img_dir
        self.data_size = data_size
        
        self.coco_obj = COCO(self.annotations_path)

        if filter_categories is None:
            self.imgs = self.coco_obj.loadImgs(self.coco_obj.getImgIds())
            self.annotations = self.coco_obj.loadAnns(self.coco_obj.getAnnIds())
            self.categoryIds = self.coco_obj.getCatIds()
            self.categories = self.coco_obj.loadCats(self.categoryIds)
            self.categoryNames = [cat["name"] for cat in self.categories]
            random.shuffle(self.imgs)
        else:
            self.filterDataset(filter_categories)
        

    def filterDataset(self, categories: list):  
        """ Filters the loaded dataset by the given list of category names.

        Args:
            categories (list): List of category names to use.
        """
        catIds = self.coco_obj.getCatIds(catNms=categories)
        imgIds = self.coco_obj.getImgIds(catIds=catIds)
        self.imgs = self.coco_obj.loadImgs(imgIds)
        self.annotations = self.coco_obj.loadAnns(self.coco_obj.getAnnIds(catIds=catIds))
        self.categoryIds = catIds
        self.categories = self.coco_obj.loadCats(catIds)
        self.categoryNames = categories
                
        random.shuffle(self.imgs)

    def getCategoryName(self, category_id):
        """ Returns the category name for a given id.

        Args:
            classId (int): Category ID to find the corresponding name for

        Returns:
            str: Category name or None if not found
        """
        for cat in self.categories:
            if cat['id']==category_id:
                return cat['name']
        return None

    def generateMask(self, img: dict):
        """ Generate the segmentation mask for a given coco image object.

        Args:
            img (dict): Coco image object to generate the segmentation mask for (Mask size: (img["height"], img["width"], 1))

        Returns:
            nparray: Segementation mask for the given image
        """
        annIds = self.coco_obj.getAnnIds(img['id'], catIds=self.categoryIds, iscrowd=None)
        anns = self.coco_obj.loadAnns(annIds)
        mask = np.zeros((img["height"], img["width"]))
        for a in range(len(anns)):
            className = self.getCategoryName(anns[a]['category_id'])
            pixel_value = self.categoryNames.index(className)+1
            mask = np.maximum(self.coco_obj.annToMask(anns[a])*pixel_value, mask)
        mask = cv2.resize(mask, (self.data_size[0], self.data_size[1]))
        mask = mask[..., np.newaxis]
        return mask

    def loadImage(self, image_obj: dict):
        """ Loads the real image for a given coco image object.

        Args:
            image_obj (dict): Coco image object to load the image for 

        Returns:
            array: The loaded color image
        """
        img = cv2.imread(os.path.join(self.img_dir, image_obj['file_name']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  / 255.0
        img = cv2.resize(img, self.data_size)
        if (len(img.shape)==3 and img.shape[2]==3):
            return img
        else:
            stacked_img = np.stack((img,)*3, axis=-1)
            return stacked_img

    def createDataGenerator(self, categories=None, batch_size=4):
        """ Creates the dataset generator with given batch size and possible filtering,

        Args:
            categories (list, optional): List of category names to use. Uses all if None. Defaults to None.
            batch_size (int, optional): Defines the size of the return batch. Defaults to 4.

        Yields:
            tuple: Batch of (images, masks) with a first dimension of "batch_size"
        """
        if categories:
            self.filterDataset(categories)

        iteration_cnt = 0
        iteration_max = len(self.imgs)
        while True:
            img_batch = np.zeros((batch_size, self.data_size[0], self.data_size[1], 3)).astype(np.float32)
            mask_batch = np.zeros((batch_size, self.data_size[0], self.data_size[1], 1)).astype(np.float32)
            for i in range(iteration_cnt, iteration_cnt+batch_size):
                img = self.imgs[i]
                img_batch[i-iteration_cnt] = self.loadImage(img)
                mask_batch[i-iteration_cnt] = self.generateMask(img)
            
            iteration_cnt += batch_size
            if iteration_cnt + batch_size >= iteration_max:
                iteration_cnt = 0
                random.shuffle(self.imgs)

            yield img_batch, mask_batch

    def visualizeGenerator(self, gen):
        """ Visualizes 4 examples of a given generator. 

        Args:
            gen (generator): Generator to call "next()" on to get data
        """
        img, mask = next(gen)
        fig = plt.figure(figsize=(20, 10))
        outerGrid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)
        
        for i in range(2):
            innerGrid = gridspec.GridSpecFromSubplotSpec(2, 2,
                            subplot_spec=outerGrid[i], wspace=0.05, hspace=0.05)

            for j in range(4):
                ax = plt.Subplot(fig, innerGrid[j])
                if(i==1):
                    ax.imshow(img[j])
                else:
                    ax.imshow(mask[j][:,:,0])
                    
                ax.axis('off')
                fig.add_subplot(ax)        
        plt.show()
