# Food volume estimation
This project aims to establish a method for estimating the volume of foods in a input image using deep learning monocular depth estimation techniques.

## Todo
- Verify that projection is working correctly
- Clean up code!
- Improve discriminator (maybe add pose as conditional input)
- Test complete adversarial model
- Add possibly other preprocessing parameters to dataframe and set creation utils in - data_utils.py (e.g. filter frames using optical flow)
- Use proper intrinsics matrix!

## Done
- Implemented resnet networks for depth and pose encoding
- Network seems to be learning
