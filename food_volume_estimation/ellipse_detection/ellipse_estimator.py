import math

import numpy as np

from food_volume_estimation.ellipse_detection.ellipse import Ellipse


class EllipseEstimator(object):
    NUM_BIN_N_ACCUMULATOR = 100
    NUM_BIN_RHO_ACCUMULATOR = 180
    MAX_MAJOR_SEMI_AXIS_LEN = 500

    def __init__(self):
        pass

    def estimate(self, ellipse_cands):
        """Estimate parameters of all ellipse candidates

        Args:
            ellipse_cands: A list of EllipseCandidate instance.

        Returns:
            A list of Ellipse instance.
        """
        ellipses = []

        for ellipse_cand in ellipse_cands:
            ellipse = self._estimate(ellipse_cand)

            if ellipse is not None:
                ellipses.append(ellipse)

        return ellipses

    def _estimate(cls, ellipse_cand):
        """Estimate ellipse parameters

        Args:
            ellipse_cand: A EllipseCandidate instance.

        Returns:
            A Ellipse instance.
        """

        seg_i, seg_j, cij, ra_ij, rb_ij, sa_ij, sb_ij = ellipse_cand.seg_pair_ij.all_params
        seg_k, _, cki, ra_ki, rb_ki, sa_ki, sb_ki = ellipse_cand.seg_pair_ki.all_params

        # Estimate ellipse center
        xc = (cij[0] + cki[0]) / 2
        yc = (cij[1] + cki[1]) / 2

        # Estimate ellipse angle (rho) and N (ratio of major and minor axis
        q_values = [
            [ra_ij, sa_ij, ra_ki, sa_ki],
            [ra_ij, sa_ij, rb_ki, sb_ki],
            [rb_ij, sb_ij, rb_ki, sb_ki],
            [rb_ij, sb_ij, ra_ki, sa_ki],
        ]

        # Estimate N and rho(angle of ellipse).
        n_acc = [0] * cls.NUM_BIN_N_ACCUMULATOR      # N \in [0, 1]
        rho_acc = [0] * cls.NUM_BIN_RHO_ACCUMULATOR  # \rho \in [-\pi/2, \pi/2]
        for i in range(4):
            q1, q2_list, q3, q4_list = q_values[i]
            for q2 in q2_list:
                for q4 in q4_list:
                    alpha = q1 * q2 - q3 * q4
                    beta = (q3 * q4 + 1) * (q1 + q2) - (q1 * q2 + 1) * (q3 + q4)
                    k_plus = (-beta + math.sqrt(beta ** 2 + 4 * alpha ** 2)) / (2 * alpha)
                    try:
                        v = ((q1 - k_plus) * (q2 - k_plus) / ((1 + q1 * k_plus) * (1 + q2 * k_plus)))
                        n_plus = math.sqrt(-v)
                    except ValueError:
                        #print('ValueError exception')
                        continue  # TODO: Avoid plus v value. Is avoiding ValueError wrong process?

                    if n_plus <= 1:
                        n = n_plus
                    else:
                        n = 1.0 / n_plus

                    if n_plus <= 1:
                        rho = math.atan(k_plus)
                    else:
                        rho = math.atan(k_plus) + math.pi / 2

                    if rho > math.pi / 2:
                        rho -= math.pi

                    try:
                        rho_bin = int((rho + math.pi / 2) / math.pi * (cls.NUM_BIN_RHO_ACCUMULATOR - 1))
                    except ValueError:
                        continue

                    try:
                        n_bin = int(n * (cls.NUM_BIN_N_ACCUMULATOR - 1))
                    except ValueError:
                        continue

                    n_acc[n_bin] += 1
                    rho_acc[rho_bin] += 1

        n = np.argmax(n_acc) / (cls.NUM_BIN_N_ACCUMULATOR - 1.0)
        rho = np.argmax(rho_acc) / (cls.NUM_BIN_RHO_ACCUMULATOR - 1.0) * math.pi - math.pi / 2  # Ellipse angle
        k = math.tan(rho)

        a_acc = [0] * (cls.MAX_MAJOR_SEMI_AXIS_LEN + 1)
        for xi, yi in np.r_[seg_i.points, seg_j.points, seg_k.points]:
            x0 = ((xi - xc) + (yi - yc) * k) / math.sqrt(k ** 2 + 1)
            y0 = (-(xi - xc) * k + (yi - yc)) / math.sqrt(k ** 2 + 1)
            ax = math.sqrt((x0 ** 2 * n ** 2 + y0 ** 2) / (n ** 2 * (k ** 2 + 1)))
            a = ax / math.cos(rho)
            try:
                a_acc[int(a)] += 1
            except (IndexError, OverflowError) as error:
                # Major semi-axis found is out of accumulator range
                continue

        a = np.argmax(a_acc)  # Major semi-axis length
        b = a * n             # Minor semi-axis length

        # Compute accuracy score
        ellipse = Ellipse(center=np.array([xc, yc], dtype=np.float32), major_len=a, minor_len=b, angle=rho)
        accuracy_score = (ellipse.count_lying_points(seg_i) + ellipse.count_lying_points(seg_j) + ellipse.count_lying_points(seg_k)) / float(seg_i.points.shape[0] + seg_j.points.shape[0] + seg_k.points.shape[0])
        ellipse.accuracy_score = accuracy_score

        return ellipse
