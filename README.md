# Food volume estimation
This project aims to establish a method for image-based food volume estimation
using deep learning monocular depth estimation techniques.

## Todo
- Keep working on point cloud volume estimation. Need to filter out outliers
  in object points because they ruin the triangulation, adding big volumes
  to where they don't exist.
- Maybe use another method for the volume estimation.
- Run tests with the EPIC-Kitchens dataset. Created a training set with
  9k frames.
- Find how to scale predicted depths to correspond to scaled depth values (m).
  Some tests show that it may not be as necessary as I think.

## Done
- Added RANSAC plate surface estimation.
- Stride 10 for set creation is ok (including optical flow filtering).
