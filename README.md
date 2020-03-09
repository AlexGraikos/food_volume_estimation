# Food volume estimation
Utilizing deep learning to estimate the food volume in an input image.


## Method
#### Depth Network Training
A depth estimation network is trained using monocular video sequences, as suggested by [Godard et al.](https://arxiv.org/pdf/1806.01260.pdf). The sequences for this purpose are obtained from the [EPIC-Kitchens](http://epic-kitchens.github.io/) dataset, which includes more than fifty hours of egocentric, food handling videos. To improve predictions, the network is afterwards fine-tuned on a [food videos dataset](https://drive.google.com/open?id=1IE1s7aWVpynwrdJFeYlJxb3n80VKqxxL), captured by smartphone cameras. 

![Depth Network Training](/assets/readme_assets/depth_train.png)

### Segmentation Network Training
A [Mask RCNN](https://arxiv.org/pdf/1703.06870.pdf) instance segmentation network is trained to predict the various food object segmentation masks. Pre-trained weights on the [COCO](http://cocodataset.org/#home) dataset are fine-tuned using the [UNIMIB2016 Food Database](http://www.ivl.disco.unimib.it/activities/food-recognition/), with individual classes aggregated into a single food class to compensate for the limited number of images. The Mask RCNN implementation is taken from [matterport](https://github.com/matterport/Mask_RCNN).

#### Volume Estimation
The food input image is passed through the depth and segmentation networks to predict the depth map and food object masks respectively. These outputs, along with the camera intrinsics, generate a point cloud on which the volume estimation is perfomed. Experimentally, there is also an option to scale the predicted depth map with a prior plate diameter. The plate detection used for this purpose is based on [this implementation](https://github.com/horiken4/ellipse-detection) of the "Very Fast Ellipse Detection for Embedded Vision Applications" algorithm.

![Volume Estimation](/assets/readme_assets/vol_est.png)


## Requirements
The code is written and tested in ```python 3.6 ```. The required pip packages for running the volume estimation script are:
```
numpy==1.16.3
pandas=0.24.2
opencv-python==4.1.0.25
scipy==1.2.1
scikit-learn==0.21.1
tensorflow==1.13.1
keras==2.2.4
h5py==2.9.0
matplotlib==3.0.3
```
To install this project, first make sure you have all the required packages
```
pip install -r requirements.txt
```
and then install using the ```setup.py``` script
```
python setup.py install
```
Training the depth estimation model also requires ```image-classifiers==0.2.1``` to import the Resnet18 model and weights.


## Training
### Depth Estimation Network
To train the depth estimation network use the ```monovideo.py``` script as:
```
monovideo.py --train --train_dataframe dataFrame.csv --config config.json 
   --batch_size B --training_epochs E --model_name name --save_per S
   --starting_weights initial_weights.h5
```
The required arguments include  a [Pandas](https://pandas.pydata.org/) dataFrame (```dataframe.csv```) containing paths to frame triplets:

curr_frame | prev_frame | next_frame
------------ | ------------- | ----------
path_to_frame_t | path_to_frame_t-1 | path_to_frame_t+1
path_to_frame_t+1 | path_to_frame_t | path_to_frame_t+2
... | ... | ... 

and a JSON file (```config.json```) that describes the training parameters:
```
{
  "name": "epic-kitchens",
  "img_size": [128, 224, 3],
  "intrinsics": [[1564.51, 0, 960], [0, 1564.51, 540], [0, 0, 1]],
  "depth_range": [0.01, 10]
}
```
The model architecture is saved in ```name.json``` when the model is instantiated whereas the model weights are saved in ```name_weights_[epoch_e/final].h5``` every ```S``` epochs and when training is complete. All outputs are stored in the ```trained_models``` directory.

The triplet-defining dataFrame can be created using the ```data_utils.py``` script as:
```
data_utils.py --create_set_df --data_source data_sources --save_target df.csv --stride S
```
where the ```data_sources``` file contains the directories in which the images are saved. For example:
```
/home/usr/food_volume_estimation/datasets/EPIC_KITCHENS_2018/frames/rgb/train/P01/P03_3/
/home/usr/food_volume_estimation/datasets/EPIC_KITCHENS_2018/frames/rgb/train/P01/P05_1/
```
You can also create a training set from multiple EPIC-Kitchens source directories, resizing the images and applying optical flow filtering ([proposed by Zhou et al.](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/cvpr17_sfm_final.pdf)) to reduce overall training costs:
```
data_utils.py --create_EPIC_set --data_source data_sources --save_target save_dir 
  --target_width W --target_height H --interp [nearest/bilinear/cubic] --stride S
  ```
To avoid redefining the data sources after creating and saving a training set of images, use the ```create_dir_df``` flag:
```
data_utils.py --create_dir_df --data_source img_dir --save_target df.csv --stride S
```
The recommended stride value for the EPIC-Kitchens dataset is 10. The extracted frames per second, used for the food videos is 4.

### Segmentation Network
To train the segmentation network use the ```food_instance_segmentation.py``` script as:
```
food_instance_segmentation.py --dataset path_to_dataset --weights starting_weights --epochs e
```
The dataset path should lead to ```train``` and ```val``` directories, containing the training and validation images. By specifying the starting weights as ```coco``` the script automatically downloads the COCO pre-trained Mask RCNN weights.


## Testing
The ```model_tests.py``` script offers testing of either all network outputs or the full-scale predicted depth:
```
model_tests.py --test_outputs --test_dataframe test_df.csv --config config.json 
  --model_architecture model_name.json --model_weights model_name_weights.h5 --n_tests 5
```
```
model_tests.py --infer_depth --test_dataframe test_df.csv --config config.json 
  --model_architecture model_name.json --model_weights model_name_weights.h5 --n_tests 5
```
Again, a Pandas dataFrame defining the frame triplets is required, since the all-outputs test generates the source to target frame reconstructions. All tests are performed without data augmentation.


## Volume Estimation
To estimate the food volume in an input image use the ```volume_estimator.py``` script as:
```
volume_estimator.py --input_images [img_path_1 img_path_2] --depth_model_architecture model_name.json
  --depth_model_weights model_name_weights.h5 --segmentation_weights segmentation_model_weights.h5
  --fov D --gt_depth_scale S --plate_diameter_prior D --min_depth min_d --max_depth max_d
  --relaxation_param relax_param --plot_results --results_file results.csv --plots_directory plots/
```
The model architecture and weights are generated by the training script, as discussed above. The camera field of view (FoV) is used to generate the intrinsics matrix during runtime. The gt_depth_scale is the expected distance between the camera and food object and is used in scaling the depth map. The min depth, max depth and relaxation parameters are model-dependent and should not be changed unless the model has been retrained and tested with these new values. If a plate diameter prior is given, then the rescaling is performed by matching the detected plate diameter with the given value. 

If you wish to visualize the volume estimation pipeline, run the example notebook ```visualize_volume_estimation.ipynb```. Point cloud plots are dependent on the [PyntCloud library](https://github.com/daavoo/pyntcloud).


## Models
Download links for the pre-trained models:
- Depth prediction model:
  - Architecture: https://drive.google.com/open?id=1t-nlUvbtD6ungcj0I_BweubRnbUogjVG
  - Weights: https://drive.google.com/open?id=1fbzlVq3KaqVBtsLNBEsEMQLnr-hc8GkB
- Segmentation model:
  - Weights: https://drive.google.com/open?id=1Y-TEatoy16QwHyRQvWBkvd0ZHd4E0Lgd


## Volume estimation examples:
Example 1 - No plate scaling | Example 2 - No plate scaling 
------------ | -------------
![Example 1](/assets/readme_assets/examples/example_pastitsio.png) | ![Example 2](/assets/readme_assets/examples/example_bread.png)

Example 3 - Plate scaling | Example 4 - Plate scaling 
------------ | -------------
![Example 3](/assets/readme_assets/examples/example_rice.png) | ![Example 4](/assets/readme_assets/examples/example_spaghetti.png)

Example 5 - Plate scaling and multiple foods  | Example 6 - Plate scaling and multiple foods
------------ | -------------
![Example 5](/assets/readme_assets/examples/example_combined_chicken.png) | ![Example 6](/assets/readme_assets/examples/example_combined_potatoes.png)


