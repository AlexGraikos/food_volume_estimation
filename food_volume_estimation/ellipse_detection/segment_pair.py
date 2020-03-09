class SegmentPair(object):

    def __init__(self, seg_a, seg_b, ellipse_center, ra, rb, sa, sb):
        """Constructor

        Args:
            seg_a: A Segment.
            seg_b: A Segment.
            ellipse_center: A numpy array of estimated ellipse center.
            ra: Slope of chord passes through seg_a extreme and seg_b middle point.
            rb: Slope of chord passes through seg_b extreme and seg_a middle point.
            sa: A list of slopes computed into the Theil-Sen estimator.
            sb: A list of slopes computed into the Theil-Sen estimator.
        """

        self._seg_a = seg_a
        self._seg_b = seg_b
        self._ellipse_center = ellipse_center
        self._ra = ra
        self._rb = rb
        self._sa = sa
        self._sb = sb

    @property
    def ellipse_center(self):
        return self._ellipse_center

    @property
    def seg_a(self):
        return self._seg_a

    @property
    def seg_b(self):
        return self._seg_b

    @property
    def all_params(self):
        """Return all parameters of this segment pair.

        Returns:
            seg_a, seg_b, ellipse_center, ra, rb, sa, sb
        """

        return self._seg_a, self._seg_a, self._ellipse_center, self._ra, self._rb, self._sa, self._sb
