# Food volume estimation
This project aims to establish a method for image-based food volume estimation
using deep learning monocular depth estimation techniques.

## Todo
- Verity network and loss functions w/ tests.
- Decide upon pose network implementation.
- Finalize EPIC-Kitchens preprocessing utilities. Network seems to be unable
  to learn from this dataset due to insufficient camera movement/non-rigid 
  object movement between frames.
- NEXT STEP: [GAN] Add discriminator network (pose as conditional input?).

## Done
- Network seems to be learning effectively on KITTI
- Fixed some issues with losses.
- Cleaned up code (data utils and tests).
