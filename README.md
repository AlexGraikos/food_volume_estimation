# Food volume estimation
This project aims to establish a method for image-based food volume estimation
using deep learning monocular depth estimation techniques.

## Todo
- Verify network and loss functions w/ tests.
- Add data augmentation preprocessing utility
- Finalize EPIC-Kitchens preprocessing utilities. Network seems to be unable
  to learn from this dataset due to insufficient camera movement/non-rigid 
  object movement between frames.
- NEXT STEP: [GAN] Add discriminator network (pose as conditional input?).

## Done
- Implemented pre-trained pose network w/ 6-channel input
- Network seems to be learning effectively on KITTI (not really)
- Fixed some issues with losses.
- Cleaned up code (data utils and tests).
