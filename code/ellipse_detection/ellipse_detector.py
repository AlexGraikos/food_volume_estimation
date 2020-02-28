import numpy as np
import cv2
from ellipse_detection.ellipse import Ellipse
from ellipse_detection.segment_detector import SegmentDetector
from ellipse_detection.ellipse_candidate_maker import EllipseCandidateMaker
from ellipse_detection.ellipse_estimator import EllipseEstimator
from ellipse_detection.ellipse_merger import EllipseMerger


class EllipseDetector(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def detect(self, image_path):
        """Detect ellipse from image.

        Args:
            image_path: Input image path

        Returns:
            Array of Ellipse instance that was detected from image.
        """

        # Load and convert image to grayscale
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (int(self.input_shape[1]),
                                   int(self.input_shape[0])))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Author's mistake??

        seg_detector = SegmentDetector()
        segments = seg_detector.detect(image)

        ellipse_cand_maker = EllipseCandidateMaker()
        ellipse_cands = ellipse_cand_maker.make(segments)

        ellipse_estimator = EllipseEstimator()
        ellipses = ellipse_estimator.estimate(ellipse_cands)

        ellipse_merger = EllipseMerger(image.shape[1],
                                       image.shape[0])
        ellipses = ellipse_merger.merge(ellipses)

        # Return the best-fitting ellipse parameters
        best_fit_ellipse = Ellipse(np.zeros(2), 0, 0, 0)
        for ellipse in ellipses:
            if ellipse.accuracy_score > best_fit_ellipse.accuracy_score:
                best_fit_ellipse = ellipse

        return (best_fit_ellipse.center[0], best_fit_ellipse.center[1],
                best_fit_ellipse.major_len, best_fit_ellipse.minor_len,
                best_fit_ellipse.angle)
