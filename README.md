# Food volume estimation
This project aims to establish a method for estimating the volume of foods in a input image using deep learning monocular depth estimation techniques.

## Todo
- scaleMinMAE is not working properly. Compare again with per-reprojection MAE and keep testing
- Verify that everything is working OK up until now
- Improve discriminator. Its sub optimal design can be vastly improved
- Test complete adversarial model
- Clean up code!
- Add possibly other preprocessing parameters to dataframe and set creation utils in - data_utils.py (e.g. filter frames using optical flow)

## Done
- Improved data preprocessing utilities - added optical flow filtering
- Added depth smoothness loss functions. Could add as a loss to model in the future
- Added per-scale min error (MAE)
- Now applying correct (95%) inverse depth normalization

