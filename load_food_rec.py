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

TRAIN_ANNOTATIONS_PATH = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_training_set_release_2.0/annotations.json"
TRAIN_IMAGE_DIRECTORY = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_training_set_release_2.0/images/"

VAL_ANNOTATIONS_PATH = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_validation_set_2.0/annotations.json"
VAL_IMAGE_DIRECTORY = "/home/jannes/Documents/MasterDelft/Q4/DeepLearning/datasets/food_rec/raw_data/public_validation_set_2.0/images/"


class CocoWrapper:

    def __init__(self, train_annotations_path: str, train_img_dir: str, val_annotations_path: str, val_img_dir: str) -> None:
        self.train_ann_path = train_annotations_path
        self.val_ann_path = val_annotations_path
        self.train_img_dir = train_img_dir
        self.val_img_dir = val_img_dir
        
        self.__load()
        self.__fix_data()

        self.loaded = False

    def __load(self):
        assert os.path.isfile(self.train_ann_path), f"Provided path for train annotations is invalid!"
        assert os.path.isfile(self.val_ann_path), f"Provided path for validation annotations is invalid!"
        assert os.path.isdir(self.train_img_dir), f"Provided path for train images is invalid!"
        assert os.path.isdir(self.val_img_dir), f"Provided path for validation images is invalid!"
        
        self.coco_obj_train = COCO(self.train_ann_path)
        self.coco_obj_validation = COCO(self.val_ann_path)

        self.train_imgs = self.coco_obj_train.loadImgs(self.coco_obj_train.getImgIds())
        self.train_annotations = self.coco_obj_train.loadAnns(self.coco_obj_train.getAnnIds())
        self.train_categories = self.coco_obj_train.loadCats(self.coco_obj_train.getCatIds())
        self.train_categorie_names = [cat["name"] for cat in self.train_categories]
        
        self.validation_imgs = self.coco_obj_validation.loadImgs(self.coco_obj_validation.getImgIds())
        self.validation_annotations = self.coco_obj_validation.loadAnns(self.coco_obj_validation.getAnnIds())
        self.validation_categories = self.coco_obj_validation.loadCats(self.coco_obj_validation.getCatIds())
        self.validation_categorie_names = [cat["name"] for cat in self.validation_categories]

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
        
    def getClassName(self, classId, coco_categories):
        for i in range(len(coco_categories)):
            if coco_categories[i]['id']==classId:
                return coco_categories[i]['name']
        return None

    def generate_masks(self, training_images):
        imgs = []
        if training_images:
            imgs = self.train_imgs
        else:
            imgs = self.validation_imgs
        masks = []
        for i in range(len(imgs))[:1]:
            img = imgs[i]
            mask = np.zeros((img['height'],img['width']))
            anns = self.coco_obj.loadAnns(self.coco_obj.getAnnIds(imgIds=img["id"]))
            for ann in anns:
                className = self.getClassName(ann['category_id'], self.categories)
                pixel_value = self.categorie_names.index(className)+1
                mask = np.maximum(self.coco_obj.annToMask(ann)*pixel_value, mask).astype(np.uint8)
            masks.append(mask)
            if i % 100 == 0:
                print(f"Annotations of Image: {i+1}/{len(imgs)}")
        return masks

def load_food_recognition_22_dataset(train_annotations_path: str, train_img_dir: str, val_annotations_path: str, val_img_dir: str):
    assert train_annotations_path is not None and os.path.isfile(train_annotations_path), f"Provided path is not valid!"

    train_coco = COCO(train_annotations_path)
    with open(train_annotations_path) as f:
        train_annotations_data = json.load(f)

    with open(val_annotations_path) as f:
        val_annotations_data = json.load(f)


    categories = train_coco.loadCats(train_coco.getCatIds())



    category_names = [_["name_readable"] for _ in categories]

    # Getting all categoriy with respective to their total images
    no_images_per_category = {}

    for n, i in enumerate(train_coco.getCatIds()):
        imgIds = train_coco.getImgIds(catIds=i)
        label = category_names[n]
        no_images_per_category[label] = len(imgIds)

    img_info = pd.DataFrame(train_coco.loadImgs(train_coco.getImgIds()))
    no_images_per_category = OrderedDict(sorted(no_images_per_category.items(), key=lambda x: -1 * x[1]))
    
    train_annotations_data = fix_data(train_annotations_data, train_img_dir)

    with open('data/train/new_ann.json', 'w') as f:
        json.dump(train_annotations_data, f)
    val_annotations_data = fix_data(val_annotations_data, val_img_dir)

    with open('data/val/new_ann.json', 'w') as f:
        json.dump(val_annotations_data, f)

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

    # # load and render the image
    # plt.imshow(plt.imread(TRAIN_IMAGE_DIRECTIORY + train_annotations_data['images'][num_images]['file_name']))
    # plt.axis('off')
    # Render annotations on top of the image
    train_coco.showAnns(annotations)
    plt.show()


if __name__ == "__main__":
    # load_food_recognition_22_dataset(TRAIN_ANNOTATIONS_PATH, TRAIN_IMAGE_DIRECTORY, VAL_ANNOTATIONS_PATH, VAL_IMAGE_DIRECTORY)
    c = CocoWrapper.createDataset(TRAIN_ANNOTATIONS_PATH, TRAIN_IMAGE_DIRECTORY, VAL_ANNOTATIONS_PATH, VAL_IMAGE_DIRECTORY)

