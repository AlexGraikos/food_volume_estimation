# Food volume estimation
This project aims to establish a method for image-based food volume estimation
using deep learning monocular depth estimation techniques.

## Todo
- Calibrate depth predictions. Find a rescaling factor that maps predicted
  depths to real-world depth values.
- Volume estimation is still debated. Fitting volume primitives based on
  food detection seems to be the best approach.
- Train the high-res model (448x256 inputs).

## Done
- Trained low-res model (224x128 inputs) and achieved promising results.
  Examples:

<p align="center">
  <b>Example 1<sup>[1]</sup></b>: <br>
  <img src="tests/point_cloud/results/test_1_depth.png" width="608">
</p>

<p align="center">
  <b>Example 1<sup>[1]</sup></b>: <br>
  <img src="tests/point_cloud/results/test_2_depth.png" width="608">
</p>

<p align="center">
  <b>Example 1<sup>[1]</sup></b>: <br>
  <img src="tests/point_cloud/results/test_3_depth.png" width="608">
</p>

<p align="center">
  <b>Example 1<sup>[1]</sup></b>: <br>
  <img src="tests/point_cloud/results/test_4_depth.png" width="608">
</p>
