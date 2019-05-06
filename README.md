# Food volume estimation
This project aims to establish a method for estimating the volume of foods in a input image using deep learning monocular depth estimation techniques.

## Todo
- Improve discriminator. Its sub optimal design can be vastly improved
- Test complete adversarial model
- Monodepth concatenates previous scale disparities to next disparity layer. Look into it
- Verify that everything is working OK up until now
- Clean up code!
- Add possibly other preprocessing parameters to dataframe and set creation utils in - data_utils.py (e.g. filter frames using optical flow)

## Done
- Implemented discriminator (single scale)
- Added multiple decoder output scales
- Added per-scale min error (MAE)
- Now applying correct (95%) inverse depth normalization

