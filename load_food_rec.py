from collections import OrderedDict
from pycocotools.coco import COCO
import json
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
import plotly.express as px
import tqdm
import cv2
import random
from tensorflow.keras import ImageDataGenerator

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


class CocoDatasetGenerator:

    def __init__(self, annotations_path: str, img_dir: str, filter_categories=None) -> None:
        assert os.path.isfile(self.annotations_path), f"Provided path for train annotations is invalid!"
        self.annotations_path = annotations_path
        self.img_dir = img_dir
        
        self.coco_obj = COCO(self.annotations_path)

        if filter_categories is not None:
            self.imgs = self.coco_obj.loadImgs(self.coco_obj.getImgIds())
            self.annotations = self.coco_obj.loadAnns(self.coco_obj.getAnnIds())
            self.categoryIds = self.coco_obj.getCatIds()
            self.categories = self.coco_obj.loadCats(self.categoryIds)
            self.categoryNames = [cat["name"] for cat in self.categories]
        else:
            self.filterDataset(filter_categories)
        
    def filterDataset(self, categories: list):        
        self.imgs = []
        self.annotations = []
        self.categories = []
        self.categoryNames = []
    
        catIds = self.coco_obj.getCatIds(catNms=categories)
        imgIds = self.coco_obj.getImgIds(catIds=catIds)
        self.imgs = self.coco_obj.loadImgs(imgIds)
        self.annotations = self.coco_obj.loadAnns(self.coco_obj.getAnnIds(catIds=catIds))
        self.categoryIds = catIds
        self.categories = self.coco_obj.loadCats(catIds)
        self.categoryNames = categories
                
        random.shuffle(self.imgs)

    def getClassName(self, classId):
        for cat in self.categories:
            if cat['id']==classId:
                return cat['name']
        return None

    def generate_mask(self, img, input_size=(224,224)):
        annIds = self.coco_obj.getAnnIds(img['id'], catIds=self.categoryIds, iscrowd=None)
        anns = self.coco_obj.loadAnns(annIds)
        mask = np.zeros((input_size))
        for a in range(len(anns)):
            className = self.getClassName(anns[a]['category_id'])
            pixel_value = self.categories.index(className)+1
            mask = np.maximum(self.coco_obj.annToMask(anns[a])*pixel_value, mask)

        return mask.reshape(input_size[0], input_size[1], 1)

    def create_data_generator(self, categories=None, input_size=(224,224), batch_size=4, augmentation=False, augmentation_args=None):
        """
        augGeneratorArgs = dict(featurewise_center = False, 
                        samplewise_center = False,
                        rotation_range = 5, 
                        width_shift_range = 0.01, 
                        height_shift_range = 0.01, 
                        brightness_range = (0.8,1.2),
                        shear_range = 0.01,
                        zoom_range = [1, 1.25],  
                        horizontal_flip = True, 
                        vertical_flip = False,
                        fill_mode = 'reflect',
                        data_format = 'channels_last')
        """
        if categories:
            self.filterDataset(categories)
        imgAugGenerator = None
        maskAugGenerator = None
        if augmentation and augmentation_args is not None:
            imgAugGenerator = ImageDataGenerator(**augmentation_args)
            maskAugGenerator = ImageDataGenerator(**augmentation_args)

        iteration_cnt = 0
        iteration_max = len(self.imgs)
        while True:
            img_batch = np.zeros((batch_size, input_size[0], input_size[1], 3)).astype(np.float32)
            mask_batch = np.zeros((batch_size, input_size[0], input_size[1], 1)).astype(np.float32)
            for i in range(iteration_cnt, iteration_cnt+batch_size):
                img = self.imgs[i]
                img_batch[i-iteration_cnt] = img
                mask_batch[i-iteration_cnt] = self.generate_mask(img, input_size=input_size)
            
            cnt += batch_size
            if cnt + batch_size >= iteration_max:
                iteration_cnt = 0
                random.shuffle(self.imgs)

            if imgAugGenerator is not None and maskAugGenerator is not None:
                seed = random.randint(0,9999)
                gen_img = imgAugGenerator.flow(img_batch, batch_size=batch_size, seed=seed, shuffle=True)
                gen_mask = imgAugGenerator.flow(mask_batch, batch_size=batch_size, seed=seed, shuffle=True)
                img_batch_aug = next(gen_img)
                mask_batch_aug = next(gen_mask)
                yield img_batch_aug, mask_batch_aug
            else:
                yield img_batch, mask_batch

    def visualize_generator(self, gen, num_examples=3):
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


def show_annotation_examples(train_coco, train_annotations_data, num_images):
    annIds = train_coco.getAnnIds(imgIds=train_annotations_data['images'][num_images]['id'])
    annotations = train_coco.loadAnns(annIds)
    train_coco.showAnns(annotations)
    plt.show()


if __name__ == "__main__":
   train_gen = CocoDatasetGenerator(TRAIN_ANNOTATIONS_PATH, TRAIN_IMAGE_DIRECTORY)
   val_gen = CocoDatasetGenerator(VAL_ANNOTATIONS_PATH, VAL_IMAGE_DIRECTORY)
