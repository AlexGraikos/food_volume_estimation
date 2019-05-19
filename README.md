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
- Fixed intrinsics matrix and pose scaling. They seem to be working now.
- Stride 2 (default) for set creation and stride 10 for frame selection show
  promising results.
