import cv2
import numpy as np

import ellipse_detection.segment
from ellipse_detection.segment import Segment


class SegmentDetector(object):
    MAIN_CLASS_13 = 1
    MAIN_CLASS_24 = 2

    CONVEXITY_UPWARD = 1
    CONVEXITY_DOWNWARD = 2

    def __init__(self):
        pass

    @classmethod
    def _trace_segment(cls, image, image_dir, footprint, segment_id, start):
        """Traces segment from start point.

        Args:
            image: A Canny edge image.
            image_dir: A gradient direction image.
            footprint: A numpy array describes wethere tracing proccess has visited the point already.
            segment_id: A label number indicating ordering segment is detected
            start: A tuple that describes point tracing proccess is begun

        Returns:
            A Segment instance.
        """

        # Find extreme
        footprint_extreme = footprint.copy()
        footprint_extreme[start[1], start[0]] = 1
        s = [start]
        while len(s) > 0:
            p = s.pop(0)

            pushed = False

            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (i, j) != (0, 0):
                        x = p[0] + j
                        y = p[1] + i

                        if x < 0 or image.shape[1] <= x or y < 0 or image.shape[0] <= y:
                            continue

                        if not footprint_extreme[y, x] and image[y, x]:
                            footprint_extreme[y, x] = 1
                            s.append((x, y))
                            pushed = True

            if not pushed:
                # Found extreme

                # Update start as found extreme
                start = p

        # Trace segment start from extreme found above

        points = []

        footprint[start[1], start[0]] = segment_id
        s = [start]
        while len(s) > 0:
            p = s.pop(0)
            points.append(p)

            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (i, j) != (0, 0):
                        x = p[0] + j
                        y = p[1] + i

                        if x < 0 or image.shape[1] <= x or y < 0 or image.shape[0] <= y:
                            continue

                        if not footprint[y, x] and image[y, x]:
                            footprint[y, x] = segment_id
                            s.append((x, y))

        points = np.asarray(points, dtype=np.float32)
        return ellipse_detection.segment.Segment(points, segment_id)

    @classmethod
    def _decide_convexity(cls, footprint, segment, main_class):
        """Decides segment convexity direction.

        Args:
            footprint: A numpy array indicating footprint is used for trace segment.
            segment: A Segment instance convexity direction is decided.
            main_class: Describes the segment is which main class. Either MAIN_CLASS_13 or MAIN_CLASS_24.

        Returns:
            Convexity direction either CONVEXITY_UPWARD or CONVEXITY_DOWNWARD.
        """

        x, y, w, h = segment.rect
        left_area, right_area = 0, 0
        for i in range(y, y + h):
            is_left = True  # Wethere point is left side against edge

            for j in range(x, x + w):
                if footprint[i, j] == segment.segment_id:
                    is_left = False
                    continue

                if is_left:
                    left_area += 1
                else:
                    right_area += 1

        if main_class == SegmentDetector.MAIN_CLASS_13:
            if left_area < right_area:
                return SegmentDetector.CONVEXITY_UPWARD
            else:
                return SegmentDetector.CONVEXITY_DOWNWARD
        elif main_class == SegmentDetector.MAIN_CLASS_24:
            if left_area < right_area:
                return SegmentDetector.CONVEXITY_DOWNWARD
            else:
                return SegmentDetector.CONVEXITY_UPWARD

    @classmethod
    def _truncate_edge_gradient_direction(cls, direction):
        if direction > np.pi / 2:
            return direction - np.pi
        elif direction < -np.pi / 2:
            return direction + np.pi

        return direction

    def detect(self, image):
        """Detects segment from image.

        Args:
            image: A gray scale image contains ellipse.

        Returns:
            A list of segments detected from image. The segments is divided into 4 classes.
        """

        # TODO: You should implement Canny edge detection for boost speed.

        # Detect edge by Canny edge detector
        image_edge = cv2.Canny(image=image, threshold1=100, threshold2=200)

        # Compute pixel gradient direction by Sovel filter
        image_gauss = cv2.GaussianBlur(src=image, ksize=(5, 5), sigmaX=3)
        image_dx = cv2.Sobel(src=image_gauss, ddepth=cv2.CV_32FC1, dx=1, dy=0)
        image_dy = cv2.Sobel(src=image_gauss, ddepth=cv2.CV_32FC1, dx=0, dy=1)

        image_dir = np.arctan2(image_dy, image_dx)
        truncate = np.vectorize(SegmentDetector._truncate_edge_gradient_direction)
        image_dir = truncate(image_dir)

        # Divide edge image to 2 class by pixel gradient direction
        image_13 = np.zeros(shape=image.shape, dtype=np.uint8)  # Edge gradient is upward to the right
        image_24 = np.zeros(shape=image.shape, dtype=np.uint8)  # Edge gradient is downward to the right
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image_edge[i, j]:
                    if image_dir[i, j] > 0:
                        image_13[i, j] = image_edge[i, j]
                    else:
                        image_24[i, j] = image_edge[i, j]

        # Detect edge segments
        segments_13 = []
        segments_24 = []

        segment_id = 1

        footprint = np.zeros(shape=image.shape, dtype=np.int)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if not footprint[i, j] and image_13[i, j]:
                    segment = SegmentDetector._trace_segment(image=image_13, image_dir=image_dir, footprint=footprint, segment_id=segment_id, start=(j, i))

                    if not segment.is_noise() and not segment.is_straight():
                        segment_id += 1
                        segments_13.append(segment)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if not footprint[i, j] and image_24[i, j]:
                    segment = SegmentDetector._trace_segment(image=image_24, image_dir=image_dir,  footprint=footprint, segment_id=segment_id, start=(j, i))

                    if not segment.is_noise() and not segment.is_straight():
                        segment_id += 1
                        segments_24.append(segment)

        # Decide edge segment convexity direction and divide 4 classes
        segments = [[], [], [], []]

        for segment in segments_13:
            convexity = SegmentDetector._decide_convexity(footprint=footprint, segment=segment, main_class=SegmentDetector.MAIN_CLASS_13)
            if convexity == SegmentDetector.CONVEXITY_UPWARD:
                segment.seg_class = Segment.SEG_CLASS_3
                segments[2].append(segment)
            elif convexity == SegmentDetector.CONVEXITY_DOWNWARD:
                segment.seg_class = Segment.SEG_CLASS_1
                segments[0].append(segment)

        for segment in segments_24:
            convexity = SegmentDetector._decide_convexity(footprint=footprint, segment=segment, main_class=SegmentDetector.MAIN_CLASS_24)
            if convexity == SegmentDetector.CONVEXITY_UPWARD:
                segment.seg_class = Segment.SEG_CLASS_4
                segments[3].append(segment)
            elif convexity == SegmentDetector.CONVEXITY_DOWNWARD:
                segment.seg_class = Segment.SEG_CLASS_2
                segments[1].append(segment)

        return segments
