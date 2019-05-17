# Food volume estimation
This project aims to establish a method for image-based food volume estimation
using deep learning monocular depth estimation techniques.

## Todo
- Run tests with the EPIC-Kitchens dataset.
- Finalize EPIC-Kitchens preprocessing utilities. Network seems to be unable
  to learn from this dataset due to insufficient camera movement/non-rigid 
  object movement between frames.
- NEXT STEP: [GAN] Add discriminator network (pose as conditional input?).

## Done
- Verified that the training flag is working and augmentations are only
  applied during the learning phase.
- Added per-scale weight to smoothness loss.
- Network seems to be learning effectively on KITTI. 
  Changing depth min-max values and intrinsics could enhance results.
