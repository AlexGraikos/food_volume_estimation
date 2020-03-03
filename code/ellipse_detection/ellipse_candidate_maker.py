import numpy as np
import cv2

from ellipse_detection.ellipse_center_estimator import EllipseCenterEstimator
from ellipse_detection.ellipse_candidate import EllipseCandidate


class EllipseCandidateMaker(object):
    ELLIPSE_CENTER_REJECT_DISTANCE = 25

    def __init__(self):
        pass

    @classmethod
    def _is_arrangement_valid(cls, seg_i, seg_j, seg_k, ccw_order):
        """Check for segment i, j and k is valid arrangement that may construct ellipse.

        Args:
            seg_i:
            seg_j:
            seg_k:
            ccw_order: A triplet indicates each class of segment i, j and k.
                       The value is one of (1,2,4), (2,3,1), (3,4,2), (4,1,3).
        """

        if ccw_order == (1, 2, 4):
            if seg_j.is_right(seg_i) and \
                    seg_k.is_up(seg_i) and \
                    seg_j.is_right(seg_k):
                        return True
        elif ccw_order == (2, 3, 1):
            if seg_j.is_up(seg_i) and \
                    seg_k.is_left(seg_i) and \
                    seg_j.is_right(seg_k):
                        return True
        elif ccw_order == (3, 4, 2):
            if seg_j.is_left(seg_i) and \
                    seg_k.is_down(seg_i) and \
                    seg_j.is_left(seg_k):
                        return True
        elif ccw_order == (4, 1, 3):
            if seg_j.is_down(seg_i) and \
                    seg_k.is_right(seg_i) and \
                    seg_j.is_left(seg_k):
                        return True

        return False

    def make(self, segments, debug_image=None):
        """Make ellipse candidate that may construct ellipse.

        Args:
            segments: A list of Segment instance.

        Returns:
            A list of EllipseCandidate.
        """

        ellipse_ce = EllipseCenterEstimator()
        ellipse_cands = []

        # Check the triplet constructed by segment (1, 2, 4)
        for seg_i in segments[0]:
            for seg_j in segments[1]:
                for seg_k in segments[3]:
                    if EllipseCandidateMaker._is_arrangement_valid(seg_i, seg_j, seg_k, (1, 2, 4)):
                        # debug
                        if debug_image is not None:
                            seg_pair_ij_image = debug_image.copy()
                            seg_pair_ki_image = debug_image.copy()

                        if debug_image is not None:
                            seg_pair_ij = ellipse_ce.estimate(seg_i, seg_j, seg_pair_ij_image)
                            seg_pair_ki = ellipse_ce.estimate(seg_k, seg_i, seg_pair_ki_image)
                        else:
                            seg_pair_ij = ellipse_ce.estimate(seg_i, seg_j)
                            seg_pair_ki = ellipse_ce.estimate(seg_k, seg_i)

                        if seg_pair_ij is None or seg_pair_ki is None:
                            continue

                        # debug
                        if debug_image is not None:
                            cv2.circle(seg_pair_ij_image, tuple(seg_pair_ij.ellipse_center.astype(np.int32)), 2, (255, 0, 0), -1)
                            cv2.circle(seg_pair_ki_image, tuple(seg_pair_ki.ellipse_center.astype(np.int32)), 2, (255, 0, 0), -1)

                        # debug
                        if debug_image is not None:
                            cv2.imshow("seg pair ij", seg_pair_ij_image)
                            cv2.imshow("seg pair ki", seg_pair_ki_image)
                            cv2.waitKey(0)

                        if np.linalg.norm(seg_pair_ij.ellipse_center - seg_pair_ki.ellipse_center) < EllipseCandidateMaker.ELLIPSE_CENTER_REJECT_DISTANCE:
                            ellipse_cand = EllipseCandidate(seg_pair_ij, seg_pair_ki)
                            ellipse_cands.append(ellipse_cand)

        # Check the triplet constructed by segment (2, 3, 1)
        for seg_i in segments[1]:
            for seg_j in segments[2]:
                for seg_k in segments[0]:
                    if EllipseCandidateMaker._is_arrangement_valid(seg_i, seg_j, seg_k, (2, 3, 1)):

                        # debug
                        if debug_image is not None:
                            seg_pair_ij_image = debug_image.copy()
                            seg_pair_ki_image = debug_image.copy()

                        if debug_image is not None:
                            seg_pair_ij = ellipse_ce.estimate(seg_i, seg_j, seg_pair_ij_image)
                            seg_pair_ki = ellipse_ce.estimate(seg_k, seg_i, seg_pair_ki_image)
                        else:
                            seg_pair_ij = ellipse_ce.estimate(seg_i, seg_j)
                            seg_pair_ki = ellipse_ce.estimate(seg_k, seg_i)

                        if seg_pair_ij is None or seg_pair_ki is None:
                            continue

                        # debug
                        if debug_image is not None:
                            cv2.circle(seg_pair_ij_image, tuple(seg_pair_ij.ellipse_center.astype(np.int32)), 2, (255, 0, 0), -1)
                            cv2.circle(seg_pair_ki_image, tuple(seg_pair_ki.ellipse_center.astype(np.int32)), 2, (255, 0, 0), -1)

                        # debug
                        if debug_image is not None:
                            cv2.imshow("seg pair ij", seg_pair_ij_image)
                            cv2.imshow("seg pair ki", seg_pair_ki_image)
                            cv2.waitKey(0)

                        if np.linalg.norm(seg_pair_ij.ellipse_center - seg_pair_ki.ellipse_center) < EllipseCandidateMaker.ELLIPSE_CENTER_REJECT_DISTANCE:
                            ellipse_cand = EllipseCandidate(seg_pair_ij, seg_pair_ki)
                            ellipse_cands.append(ellipse_cand)

        # Check the triplet constructed by segment (3, 4, 2)
        for seg_i in segments[2]:
            for seg_j in segments[3]:
                for seg_k in segments[1]:
                    if EllipseCandidateMaker._is_arrangement_valid(seg_i, seg_j, seg_k, (3, 4, 2)):
                        # debug
                        if debug_image is not None:
                            seg_pair_ij_image = debug_image.copy()
                            seg_pair_ki_image = debug_image.copy()

                        if debug_image is not None:
                            seg_pair_ij = ellipse_ce.estimate(seg_i, seg_j, seg_pair_ij_image)
                            seg_pair_ki = ellipse_ce.estimate(seg_k, seg_i, seg_pair_ki_image)
                        else:
                            seg_pair_ij = ellipse_ce.estimate(seg_i, seg_j)
                            seg_pair_ki = ellipse_ce.estimate(seg_k, seg_i)

                        if seg_pair_ij is None or seg_pair_ki is None:
                            continue

                        # debug
                        if debug_image is not None:
                            cv2.circle(seg_pair_ij_image, tuple(seg_pair_ij.ellipse_center.astype(np.int32)), 2, (255, 0, 0), -1)
                            cv2.circle(seg_pair_ki_image, tuple(seg_pair_ki.ellipse_center.astype(np.int32)), 2, (255, 0, 0), -1)

                        # debug
                        if debug_image is not None:
                            cv2.imshow("seg pair ij", seg_pair_ij_image)
                            cv2.imshow("seg pair ki", seg_pair_ki_image)
                            cv2.waitKey(0)

                        if np.linalg.norm(seg_pair_ij.ellipse_center - seg_pair_ki.ellipse_center) < EllipseCandidateMaker.ELLIPSE_CENTER_REJECT_DISTANCE:
                            ellipse_cand = EllipseCandidate(seg_pair_ij, seg_pair_ki)
                            ellipse_cands.append(ellipse_cand)

        # Check the triplet constructed by segment (4, 1, 3)
        for seg_i in segments[3]:
            for seg_j in segments[0]:
                for seg_k in segments[2]:
                    if EllipseCandidateMaker._is_arrangement_valid(seg_i, seg_j, seg_k, (4, 1, 3)):
                        # debug
                        if debug_image is not None:
                            seg_pair_ij_image = debug_image.copy()
                            seg_pair_ki_image = debug_image.copy()

                        if debug_image is not None:
                            seg_pair_ij = ellipse_ce.estimate(seg_i, seg_j, seg_pair_ij_image)
                            seg_pair_ki = ellipse_ce.estimate(seg_k, seg_i, seg_pair_ki_image)
                        else:
                            seg_pair_ij = ellipse_ce.estimate(seg_i, seg_j)
                            seg_pair_ki = ellipse_ce.estimate(seg_k, seg_i)

                        if seg_pair_ij is None or seg_pair_ki is None:
                            continue

                        # debug
                        if debug_image is not None:
                            cv2.circle(seg_pair_ij_image, tuple(seg_pair_ij.ellipse_center.astype(np.int32)), 2, (255, 0, 0), -1)
                            cv2.circle(seg_pair_ki_image, tuple(seg_pair_ki.ellipse_center.astype(np.int32)), 2, (255, 0, 0), -1)

                        # debug
                        if debug_image is not None:
                            cv2.imshow("seg pair ij", seg_pair_ij_image)
                            cv2.imshow("seg pair ki", seg_pair_ki_image)
                            cv2.waitKey(0)

                        if np.linalg.norm(seg_pair_ij.ellipse_center - seg_pair_ki.ellipse_center) < EllipseCandidateMaker.ELLIPSE_CENTER_REJECT_DISTANCE:
                            ellipse_cand = EllipseCandidate(seg_pair_ij, seg_pair_ki)
                            ellipse_cands.append(ellipse_cand)

        return ellipse_cands
