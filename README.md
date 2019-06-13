# Food volume estimation
Using monocular depth estimation to estimate food volume in an input image.

## Method
#### Depth Network Training
A monocular depth estimation network is trained using monocular video sequences, as suggested by [Godard et al.](https://arxiv.org/pdf/1806.01260.pdf) . The sequences used for this purpose are obtained from the [EPIC-Kitchens](http://epic-kitchens.github.io/) dataset, which includes many hours of egocentric, food handling videos. 

![Depth Network Training](/sources/depth_train.png)

#### Volume Estimation
The food input image is passed through the trained depth network and the predicted depth map, along with the camera intrinsics, are used to generate a corresponding point cloud.

![Volume Estimation](/sources/vol_est.png)


## Training
To train the depth estimation network use the ```monovideo.py``` script as:
```
monovideo.py --train --train_dataframe dataFrame.csv --config config.json 
   --batch_size B --training_epochs E --model_name name --save_per S
```
The required arguments include  a [Pandas](https://pandas.pydata.org/) dataFrame (```dataframe.csv```) containing paths to frame triplets:

curr_frame | prev_frame | next_frame
------------ | ------------- | ----------
path_to_frame_t | path_to_frame_t-1 | path_to_frame_t+1
path_to_frame_t+1 | path_to_frame_t | path_to_frame_t+2
... | ... | ... 

and a JSON file (```config.json```) that describes various training parameters:
```
{
  "name": "epic-kitchens",
  "img_size": [128, 224, 3],
  "intrinsics": [[1564.51, 0, 960], [0, 1564.51, 540], [0, 0, 1]],
  "depth_range": [0.01, 10]
}
```
The model architecture is saved in ```name.json``` when the model is instantiated whereas the model weights are saved in ```name_weights_[epoch_e/final].h5``` every ```S``` epochs and when training is complete ([H5 format](https://www.h5py.org/)). All outputs are stored in the ```trained_models``` directory.

The triplet-defining dataFrame can be created using the ```data_utils.py``` script as:
```
data_utils.py --create_set_df --data_source data_sources --save_target df.csv --stride S
```
where the ```data_sources file``` contains the directories in which the images are saved. For example:
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
The recommended stride value for the EPIC-Kitchens dataset is 10.

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
Currently the volume estimation algorithm is implemented in ```tests/point_cloud/infer_depth_and_estimate_volume.ipynb``` for visualization and testing purposes. It is subject to change. 


## Todo
- [x] Trained low-res model (224x128 inputs) and achieved promising results.
  Examples:
- [ ] Calibrate depth predictions. Find a rescaling factor that maps predicted
  depths to real-world depth values.
- [ ] Volume estimation is still debated. Fitting volume primitives based on
  food detection seems to be the best approach.
- [ ] Train the high-res model (448x256 inputs).
  
Low-res model examples:

Example 1 | Example 2
------------ | -------------
![Example 1](/tests/point_cloud/results/test_1_depth.png) | ![Example 2](/tests/point_cloud/results/test_2_depth.png)

Example 3 | Example 4
------------ | -------------
![Example 1](/tests/point_cloud/results/test_1_depth.png) | ![Example 2](/tests/point_cloud/results/test_2_depth.png)
</table> 

