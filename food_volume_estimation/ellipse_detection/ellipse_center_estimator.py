import numpy as np
import cv2

from food_volume_estimation.ellipse_detection.segment import Segment
from food_volume_estimation.ellipse_detection.segment_pair import SegmentPair


class EllipseCenterEstimator(object):
    def __init__(self):
        self._seg_pair_cache = {}

    def estimate(self, seg_a, seg_b, debug_image=None):
        """Estimate ellipse center indicated by segmtens and slopes of chords and lines pass through chord midpoints.

        Args:
            seg_a: A Segment.
            seg_b: A Segment.

        Returns:
            A SegmentPair or None if middle points too few.
        """

        # If segment pair cache exists, return it
        cache_key = str(seg_a.segment_id) + '-' + str(seg_b.segment_id)
        if cache_key in self._seg_pair_cache:
            return self._seg_pair_cache[cache_key]

        # debug
        if debug_image is not None:
            seg_ab_midpoints_image = debug_image.copy()
            seg_ba_midpoints_image = debug_image.copy()

        if debug_image is not None:
            midpoints, ra, ma = EllipseCenterEstimator._compute_chord_midpoints(seg_a, seg_b, seg_ab_midpoints_image)
        else:
            midpoints, ra, ma = EllipseCenterEstimator._compute_chord_midpoints(seg_a, seg_b)

        if midpoints.shape[0] <= 2:
            # Because midpoint of chords is too few ellipse center can not be estimated
            return None

        ta, sa = EllipseCenterEstimator._estimate_slope(midpoints)

        if ta == float("inf"):
            # Slope of chord is infinity
            # TODO: You should handle infinity slope
            return None

        # debug
        if debug_image is not None:
            pt1 = ma
            pt2 = ma + np.array([50, 50 * ta])
            cv2.line(seg_ab_midpoints_image, tuple(pt1.astype(np.int32)), tuple(pt2.astype(np.int32)), (255, 0, 0))

        if debug_image is not None:
            midpoints, rb, mb = EllipseCenterEstimator._compute_chord_midpoints(seg_b, seg_a, seg_ba_midpoints_image)
        else:
            midpoints, rb, mb = EllipseCenterEstimator._compute_chord_midpoints(seg_b, seg_a)

        if midpoints.shape[0] <= 2:
            # Because midpoint of chords is too few ellipse center can not be estimated
            return None

        tb, sb = EllipseCenterEstimator._estimate_slope(midpoints)

        if tb == float("inf"):
            # Slope of chord is infinity
            # TODO: You should handle infinity slope
            return None

        # debug
        if debug_image is not None:
            pt1 = mb
            pt2 = mb + np.array([50, 50 * tb])
            cv2.line(seg_ba_midpoints_image, tuple(pt1.astype(np.int32)), tuple(pt2.astype(np.int32)), (255, 0, 0))

        # debug
        if debug_image is not None:
            cv2.imshow('seg a b midpoints', seg_ab_midpoints_image)
            cv2.imshow('seg b a midpoints', seg_ba_midpoints_image)
            cv2.waitKey(0)

        # Compute ellipse center (Paper may be wrong)
        cab_x = (mb[1] - tb * mb[0] - ma[1] + ta * ma[0]) / (ta - tb)
        cab_y = (ta * mb[1] - tb * ma[1] + ta * tb * (ma[0] - mb[0])) / (ta - tb)

        seg_pair = SegmentPair(seg_a, seg_b, np.array([cab_x, cab_y]), ra, rb, sa, sb)

        # Update segment pair cache
        cache_key = str(seg_a.segment_id) + '-' + str(seg_b.segment_id)
        self._seg_pair_cache[cache_key] = seg_pair
        cache_key = str(seg_b.segment_id) + '-' + str(seg_a.segment_id)
        self._seg_pair_cache[cache_key] = SegmentPair(seg_b, seg_a, np.array([cab_x, cab_y]), rb, ra, sb, sa)

        return seg_pair

    @classmethod
    def _compute_chord_midpoints(cls, seg_a, seg_b, debug_image=None):
        """Compute midpoints of chord between two segments

        Args:
            seg_a: A Segment.
            seg_b: A Segment.

        Returns:
            midpoints, ra, ma
            midpoints: Midpoints of chords between seg_a, and seg_b.
            ra: Slope of chord.
            ma: Median point of midpoints of chords.
        """
        ha = seg_a.middle_point
        hb = seg_b.middle_point

        # Decide side of extreme
        dist_to_source = np.linalg.norm(hb - seg_a.source_extreme)
        dist_to_terminal = np.linalg.norm(hb - seg_a.terminal_extreme)
        if dist_to_source > dist_to_terminal:
            seg_a_sample_from = Segment.SAMPLE_FROM_SOURCE_EXTREME
            pa = seg_a.source_extreme
        else:
            seg_a_sample_from = Segment.SAMPLE_FROM_TERMINAL_EXTREME
            pa = seg_a.terminal_extreme

        #pa = seg_a.source_extreme
        #ha = seg_a.middle_point
        #hb = seg_b.middle_point
        pha = hb - pa

        pha_unit = pha / np.linalg.norm(pha)
        pha_normal = np.array([pha_unit[1], -pha_unit[0]])

        if np.dot(pha_normal, ha - pa) < 0:  # FIXME: Not valid ?
            pha_normal *= -1

        # Midpoints of chords has not been found
        if hb[0] == pa[0]:
            # Can not ompute slope of chord
            return np.array([]), 0, 0

        # Calculate slope of line trough the extreme and the midpoint of the segment
        ra = (hb[1] - pa[1]) / (hb[0] - pa[0])  # Paper may be wrong

        # Compute midpoints of parallel chords
        samples_a = seg_a.sample_chord_points(sample_from=seg_a_sample_from, chord_normal=pha_normal, interval=3)
        samples_b = seg_b.sample_chord_points(sample_from=Segment.SAMPLE_FROM_MIDDLE_POINT, chord_normal=pha_normal, interval=3)

        # debug
        if debug_image is not None:
            pt1 = (pa + hb) / 2
            pt2 = pt1 + pha_normal * 100
            cv2.line(debug_image, tuple(pt1.astype(np.int32)), tuple(pt2.astype(np.int32)), (255, 0, 0))
            cv2.line(debug_image, tuple(pa.astype(np.int32)), tuple(hb.astype(np.int32)), (255, 0, 0))

        # debug
        if debug_image is not None:
            seg_a.draw(debug_image)
            seg_b.draw(debug_image)

        num_midpoints = min(samples_a.shape[0], samples_b.shape[0])

        if num_midpoints == 0:
            # Midpoints of chords has not been found
            return np.array([]), 0, 0

        midpoints = np.zeros(shape=(num_midpoints, 2))
        for i in range(num_midpoints):
            a = samples_a[i]
            b = samples_b[i]

            midpoints[i] = (a + b) / 2

            # debug
            if debug_image is not None:
                cv2.line(debug_image, tuple(a.astype(np.int32)), tuple(b.astype(np.int32)), (0, 0, 255))
            #cv2.circle(debug_image, tuple(a.astype(np.int32)), 1, (255, 0, 0), -1)
            #cv2.circle(debug_image, tuple(b.astype(np.int32)), 1, (255, 0, 0), -1)
            cv2.circle(debug_image, tuple(midpoints[i].astype(np.int32)), 2, (255, 0, 0), -1)

        # Calculate median point of midpoints of parallel chords
        ma = np.median(midpoints, axis=0)

        # debug
        if debug_image is not None:
            cv2.circle(debug_image, tuple(ma.astype(np.int32)), 2, (0, 255, 0), -1)
            pt1 = ma.astype(np.int32)
            pt2 = (pt1 + np.array([50, ra * 50])).astype(np.int32)
            cv2.line(debug_image, tuple(pt1), tuple(pt2), (0, 255, 0))

        return midpoints, ra, ma

    @classmethod
    def _estimate_slope(cls, midpoints):
        """Estimate slope of the line through midpoints of chord between two segments

        Args:
            midpoints: Midpoints of chords.

        Returns:
            ta, sa
            ta: Slope of line passes through midpoints of chords.
            sa: A list of slopes computed into the Theil-Sen estimator.
        """

        slopes = []

        middle = midpoints.shape[0] // 2
        for i in range(middle):
            x1, y1 = midpoints[i]
            x2, y2 = midpoints[middle + i]
            if x2 == x1:
                slope = float("inf")
            else:
                slope = (y2 - y1) / (x2 - x1)  # Paper may be wrong
            slopes.append(slope)

        return np.median(slopes), slopes
