import numpy as np
import cv2


class Segment(object):
    SAMPLE_FROM_SOURCE_EXTREME = 1
    SAMPLE_FROM_MIDDLE_POINT = 2
    SAMPLE_FROM_TERMINAL_EXTREME = 3

    SEG_CLASS_1 = 1
    SEG_CLASS_2 = 2
    SEG_CLASS_3 = 3
    SEG_CLASS_4 = 4

    def __init__(self, points, segment_id):
        """Constructor

        Args:
            points: A numpy array
            segment_id: A value indicating ordering segment was detected
        """

        self._points = points
        self._rot_rect = cv2.minAreaRect(points)  # ((center_x, center_y), (width, height), angle)
        self._rect = cv2.boundingRect(points)  # (x, y, width, height)
        self._seg_class = None
        self._segment_id = segment_id

    @property
    def points(self):
        return self._points

    @property
    def rect(self):
        return self._rect

    @property
    def source_extreme(self):
        return self._points[0]

    @property
    def terminal_extreme(self):
        return self._points[-1]

    @property
    def middle_point(self):
        return self._points[self._points.shape[0] // 2]

    @property
    def seg_class(self):
        return self._seg_class

    @seg_class.setter
    def seg_class(self, value):
        self._seg_class = value

    @property
    def segment_id(self):
        return self._segment_id

    @segment_id.setter
    def segment_id(self, value):
        self._segment_id = value

    def is_noise(self):
        if self._points.shape[0] < 15:
            return True

        return False

    def is_straight(self):
        w, h = self._rot_rect[1]

        if w == 0 or h == 0:
            return True

        ratio = max(float(w), float(h)) / min(w, h)
        if ratio > 10:
            return True

        return False

    def sample_chord_points(self, sample_from, chord_normal, interval):
        sample_points = []

        if sample_from == Segment.SAMPLE_FROM_SOURCE_EXTREME:
            istart = 0
            istep = 1
        elif sample_from == Segment.SAMPLE_FROM_MIDDLE_POINT:
            istart = self._points.shape[0] // 2
            if np.dot(self.middle_point - self.source_extreme, chord_normal) > 0:
                istep = 1
            else:
                istep = -1
        elif sample_from == Segment.SAMPLE_FROM_TERMINAL_EXTREME:
            istart = self._points.shape[0] - 1
            istep = -1
        else:
            raise Exception('Invalid value of sample from')

        start = self._points[istart]
        for p in self._points[istart + istep::istep]:
            dist = np.dot((p - start), chord_normal)
            if dist >= interval:
                start = p
                sample_points.append(tuple(p))

        return np.asarray(sample_points, dtype=np.float32)

    def is_right(self, seg):
        """Check relative position whether seg is right side of this.

        Args:
            seg:

        Returns:
            True if seg is right side of this, otherwise False.
        """

        if self.rect[0] + self.rect[2] <= seg.rect[0]:
            return True

        return False

    def is_left(self, seg):
        """Check relative position whether seg is left side of this.

        Args:
            seg:

        Returns:
            True if seg is left side of this, otherwise False.
        """

        if seg.rect[0] + seg.rect[2] <= self.rect[0]:
            return True

        return False

    def is_up(self, seg):
        """Check relative position whether seg is up side of this.

        Args:
            seg:

        Returns:
            True if seg is up side of this, otherwise False.
        """

        if self.rect[1] + self.rect[3] <= seg.rect[1]:
            return True

        return False

    def is_down(self, seg):
        """Check relative position whether seg is down side of this.

        Args:
            seg:

        Returns:
            True if seg is down side of this, otherwise False.
        """

        if seg.rect[1] + seg.rect[3] <= self.rect[1]:
            return True

        return False

    def draw(self, image, label=None):
        # Segment edge points
        n = self._points.shape[0]
        for i, p in enumerate(self._points):
            cv2.circle(image, tuple(p), 1, ((1 - float(i) / n) * 255, 0, float(i) / n * 255))

        # Bounding box
        pt1 = (self._rect[0], self._rect[1])
        pt2 = (self._rect[0] + self._rect[2], self._rect[1] + self._rect[3])
        cv2.rectangle(image, pt1, pt2, (255, 0, 0))

        # Rotated bounding box
        vertices = cv2.boxPoints(self._rot_rect)
        begin = vertices[-1]
        for end in vertices:
            cv2.line(image, tuple(begin), tuple(end), (0, 255, 0))
            begin = end

        # Line through extremes
        pt1 = tuple(self._points[0])
        pt2 = tuple(self._points[-1])
        cv2.line(image, pt1, pt2, (255, 255, 0), 4)

        # Middle point
        cv2.circle(image, tuple(self.middle_point), 3, (0, 255, 255))

        # Extreme
        cv2.circle(image, tuple(self.source_extreme), 3, (255, 0, 255))
        cv2.circle(image, tuple(self.terminal_extreme), 3, (255, 0, 255))

        # Segment class
        cv2.putText(image, str(self._seg_class), tuple(self.middle_point), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        # Label
        if label is not None:
            cv2.putText(image, label, tuple(self.middle_point), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
