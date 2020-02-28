import numpy as np
import cv2


class EllipseCandidate(object):

    def __init__(self, seg_pair_ij, seg_pair_ki):
        """Constructor

        Args:
            seg_pari_ij: A SegmentPair.
            seg_pair_ki: A SegmentPair.
        """

        self._seg_pair_ij = seg_pair_ij
        self._seg_pair_ki = seg_pair_ki

    @property
    def seg_pair_ij(self):
        return self._seg_pair_ij

    @property
    def seg_pair_ki(self):
        return self._seg_pair_ki

    def draw(self, image):
        self._seg_pair_ij.seg_a.draw(image, 'i')
        self._seg_pair_ij.seg_b.draw(image, 'j')
        self._seg_pair_ki.seg_a.draw(image, 'k')

        cv2.circle(image, tuple(self._seg_pair_ij.ellipse_center.astype(np.int32)), 2, (0, 255, 0), -1)
        cv2.circle(image, tuple(self._seg_pair_ki.ellipse_center.astype(np.int32)), 2, (0, 255, 0), -1)
