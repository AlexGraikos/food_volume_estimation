# Food volume estimation
This project aims to establish a method for estimating the volume of foods in a input image using deep learning monocular depth estimation techniques.

## Todo
- Check why model outputs are zero. Model needs a lot of modifications
- Check encoder->decoder skip connections
- Validate that python3 port is working properly (check divisions -> they are now altered)
- Clean up code!
- Add possibly other preprocessing parameters to dataframe and set creation utils in - data_utils.py
- Run loads of tests

## Done
- Implemented depth-predicting model
- Model can be trained, no actual tests ran
- ResNet18 instead of DenseNet used to reduce computational and memory costs
- Code in model and data_utils made prettier using cmd arguments
- Ported everything to python3

