from collections import OrderedDict
from pycocotools.coco import COCO
import json
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import tqdm
import cv2
import random

TRAIN_ANNOTATIONS_PATH = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_training_set_release_2.0/annotations.json"
TRAIN_IMAGE_DIRECTORY = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_training_set_release_2.0/images/"

VAL_ANNOTATIONS_PATH = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_validation_set_2.0/annotations.json"
VAL_IMAGE_DIRECTORY = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_validation_set_2.0/images/"


class CocoWrapper:

    def __init__(self, annotations_path: str, img_dir: str) -> None:
        self.annotations_path = annotations_path
        self.img_dir = img_dir
        
        self.__load()
        self.__fix_data()

        self.loaded = False

    def __load(self):
        assert os.path.isfile(self.annotations_path), f"Provided path for train annotations is invalid!"
        assert os.path.isdir(self.img_dir), f"Provided path for train images is invalid!"
        
        self.coco_obj = COCO(self.annotations_path)

        self.imgs = self.coco_obj.loadImgs(self.coco_obj.getImgIds())
        self.annotations = self.coco_obj.loadAnns(self.coco_obj.getAnnIds())
        self.categories = self.coco_obj.loadCats(self.coco_obj.getCatIds())
        self.categorie_names = [cat["name"] for cat in self.categories]

    def fix_uneven_data(annotations: list, out_annotation_file: str, image_dir: str):
        assert os.path.isdir(image_dir), f"Fix annotations: Provided path for images is invalid!"
        assert annotations is not None, f"Fix annotations: Given annotations are None!"

        for n, i in enumerate(tqdm((annotations['images']))):
            img = cv2.imread(image_dir+i["file_name"])
            if img.shape[0] != i['height']:
                annotations['images'][n]['height'] = img.shape[0]
            if img.shape[1] != i['width']:
                annotations['images'][n]['width'] = img.shape[1]
        with open(out_annotation_file, 'w') as f:
            json.dump(annotations, f)
        
    def filterDataset(self, categories: list):        
        self.imgs = []
        self.annotations = []
        self.categories = []
        self.categorie_names = []
    
        catIds = self.coco_obj.getCatIds(catNms=categories)
        imgIds = self.coco_obj.getImgIds(catIds=catIds)
        self.imgs = self.coco_obj.loadImgs(imgIds)
        self.annotations = self.coco_obj.loadAnns(self.coco_obj.getAnnIds(catIds=catIds))
        self.categories = self.coco_obj.loadCats(catIds)
        self.categorie_names = categories
                
        random.shuffle(self.imgs)

    def getClassName(self, classId, coco_categories):
        for i in range(len(coco_categories)):
            if coco_categories[i]['id']==classId:
                return coco_categories[i]['name']
        return None

    def generate_mask(self, img, catIds, training, categories):
        # imgs = []
        # if training:
        #     imgs = self.train_imgs
        #     catIds = self.coco_obj_train.getCatIds(catNms=categories)
        # else:
        #     imgs = self.validation_imgs
        #     catIds = self.coco_obj_validation.getCatIds(catNms=categories)
        # masks = []
        # for i in range(len(imgs))[:1]:
        #     img = imgs[i]
        #     mask = np.zeros((img['height'],img['width']))
        #     anns = self.coco_obj.loadAnns(self.coco_obj.getAnnIds(imgIds=img["id"], catIds=catIds))
        #     for ann in anns:
        #         className = self.getClassName(ann['category_id'], categories)
        #         pixel_value = self.categorie_names.index(className)+1
        #         mask = np.maximum(self.coco_obj.annToMask(ann)*pixel_value, mask).astype(np.uint8)
        #     mask.reshape((img['height'],img['width'], 1))
        #     masks.append(mask)
        #     if i % 100 == 0:
        #         print(f"Annotations of Image: {i+1}/{len(imgs)}")
        # return masks
        if training:
            coco = self.coco_obj
        else:
            coco = self.coco_obj_validation

        annIds = coco.getAnnIds(img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        cats = coco.loadCats(catIds)
        train_mask = np.zeros((img['height'],img['width']))
        for a in range(len(anns)):
            className = self.getClassName(anns[a]['category_id'], cats)
            pixel_value = categories.index(className)+1
            train_mask = np.maximum(coco.annToMask(anns[a])*pixel_value, train_mask)

        return train_mask.reshape(img['height'],img['width'], 1)

    def create_data_generator(self, batch_size, train=True, classes=[]):
        cnt = 0
        while True:
            for i in range(batch_size):
                img = 

            cnt += batch_size

# Function for taking a annotation & directiory of images and returning new annoation json with fixed image size info
def fix_data(annotations, directiory):
  for n, i in enumerate(tqdm((annotations['images']))):
      img = cv2.imread(directiory+i["file_name"])
      if img.shape[0] != i['height']:
          annotations['images'][n]['height'] = img.shape[0]
        #   print(i["file_name"])
        #   print(annotations['images'][n], img.shape)

      if img.shape[1] != i['width']:
          annotations['images'][n]['width'] = img.shape[1]
        #   print(i["file_name"])
        #   print(annotations['images'][n], img.shape)
  return annotations


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
    # load_food_recognition_22_dataset(TRAIN_ANNOTATIONS_PATH, TRAIN_IMAGE_DIRECTORY, VAL_ANNOTATIONS_PATH, VAL_IMAGE_DIRECTORY)
    c = CocoWrapper.createDataset(TRAIN_ANNOTATIONS_PATH, TRAIN_IMAGE_DIRECTORY, VAL_ANNOTATIONS_PATH, VAL_IMAGE_DIRECTORY)

