import math

import numpy as np


class EllipseMerger(object):
    """Merge ellipses estimated according to similarity.

    This class adopts algorithm described by "Clustering of Ellipses based on their Distinctiveness: An Aid to Ellipse Detection Algorithms"

    Attributes:
        _image_w: A width of image ellipse was detected.
        _image_h: A height of image ellipse was detected.
    """

    IDENTIFY_THRESHOLD = 0.1

    def __init__(self, image_w, image_h):
        self._image_w = image_w
        self._image_h = image_h

    def _is_same(self, ellipse1, ellipse2):
        """Check if ellipses are same.

        Args:
            ellipse1: A Ellipse instance.
            ellipse2: A Ellipse instance.

        Returns:
            If ellipses are same return true, otherwise false
        """

        dist_center = np.abs(ellipse1.center - ellipse2.center)

        dist = np.array([1.0] * 5)
        dist[0] = dist_center[0] / self._image_w  # Distance of x
        dist[1] = dist_center[1] / self._image_h  # Distance of y

        dist[2] = abs(float(ellipse1.major_len - ellipse2.major_len)) / max(ellipse1.major_len, ellipse2.major_len)  # Distance of a
        dist[3] = abs(float(ellipse1.minor_len - ellipse2.minor_len)) / min(ellipse1.minor_len, ellipse2.minor_len)  # Distance of b

        # Distance of theta
        r1 = ellipse1.minor_len / ellipse1.major_len
        r2 = ellipse2.minor_len / ellipse2.major_len
        if r1 >= 0.9 and r2 >= 0.9:
            dist[4] = 0
        elif r1 >= 0.9 and r2 < 0.9:
            dist[4] = 1
        elif r1 < 0.9 and r2 >= 0.9:
            dist[4] = 1
        elif r1 < 0.9 and r2 < 0.9:
            dist[4] = abs(ellipse1.angle - ellipse2.angle) / math.pi

        return np.all(dist < EllipseMerger.IDENTIFY_THRESHOLD)

    def merge(self, ellipses):
        """Merge ellipse and return merged ellipses.

        Args:
            ellipses: A list of Ellipse instance.

        Returns:
            A list of merged ellipses.
        """

        if len(ellipses) == 0:
            return []

        merged_ellipses = [ellipses[0]]

        for i in range(1, len(ellipses)):
            ellipse = ellipses[i]

            merged = False
            for j in range(len(merged_ellipses)):
                merged_ellipse = merged_ellipses[j]

                if self._is_same(ellipse, merged_ellipse):
                    merged = True

                    if ellipse.accuracy_score > merged_ellipse.accuracy_score:
                        merged_ellipses[j] = ellipse

                    break

            if not merged:
                merged_ellipses.append(ellipse)

        return merged_ellipses
