import math

import numpy as np
import cv2


class Ellipse(object):
    def __init__(self, center, major_len, minor_len, angle):
        """Constructor.

        Args:
            center: numpy array indicating center of ellipse.
            major_len: Major semi-axis length.
            minor_len: Minor semi-axis length.
            angle: Angle of ellipse.
        """

        self._center = center
        self._major_len = major_len
        self._minor_len = minor_len
        self._angle = angle
        self._accuracy_score = 0

    def __str__(self):
        return '{{center: {0}, major_len: {1}, minor_len: {2}, angle: {3}, accuracy_score: {4}}}'.format(self._center, self._major_len, self._minor_len, self._angle, self._accuracy_score)

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, value):
        self._center = value

    @property
    def major_len(self):
        return self._major_len

    @property
    def minor_len(self):
        return self._minor_len

    @property
    def angle(self):
        return self._angle

    @property
    def accuracy_score(self):
        return self._accuracy_score

    @accuracy_score.setter
    def accuracy_score(self, value):
        self._accuracy_score = value

    @classmethod
    def _is_lying(cls, x):
        return 0.9 <= x <= 1.1

    def count_lying_points(self, seg):
        """Check if points is lying this ellise.

        Args:
            seg: A segment.

        Return:
            Number of points that is lying this ellipse.
        """

        # TODO: Use numpy broadcast technique

        num_lying = 0

        a = self._major_len
        b = self._minor_len
        c = math.cos(self._angle)
        s = math.sin(self._angle)
        for p in seg.points:
            x = p[0] - self._center[0]
            y = p[1] - self._center[1]

            v = (x * c + y * s) ** 2 / a ** 2 + (-x * s + y * c) ** 2 / b ** 2

            if Ellipse._is_lying(v):
                num_lying += 1

        return num_lying

    def draw(self, image):
        cv2.ellipse(image, tuple(self._center.astype(np.int32)), (int(self._major_len), int(self._minor_len)), self._angle * 180 / math.pi, 0, 360, (68 / 255, 1 / 255, 84 / 255), 2)
