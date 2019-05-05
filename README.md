# Food volume estimation
This project aims to establish a method for estimating the volume of foods in a input image using deep learning monocular depth estimation techniques.

## Todo
- Implement discriminator (single or multi scale)
- Add more decoder output scales
- (Possibly) Use per-scale min error (MAE)
- Verify that everything is working OK up until now
- Clean up code!
- Add possibly other preprocessing parameters to dataframe and set creation utils in - data_utils.py (e.g. filter frames using optical flow)

## Done
- Model now uses the encoder-decoder from adversarial monocular depth estimation
- Training with L1 and single scale yields better results than ever achieved before
- Now applying correct (90%) inverse depth normalization

