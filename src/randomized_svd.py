import numpy as np
from fps_sampling import FpsSampling
from kernel_approximation import KernelApproximation
import utils


class RandomizedSVD(FpsSampling, KernelApproximation):

    def randomized_svd(self, data, num_points_to_sample, sigma, projection_dim):
        """
        Compute all steps of randomized svd.
        Based on Facebook Research overview: https://research.fb.com/blog/2014/09/fast-randomized-svd/
        :param data: the data (num_samples, point_dim)
        :param num_points_to_sample: number of points to take
        :param sigma: the distance scale hyper-parameter
        :param projection_dim: the new dimension on which we project the kernel
        :return: U, s, Vh
        """

        # FPS sampling:
        farthest_idx = self.fps_sampling(point_array=data, num_points_to_sample=num_points_to_sample)

        # Calc C and U:
        C, U = self.calc_C_U(data=data, farthest_idx=farthest_idx, sigma=sigma)

        # Scale the rows of C (so they will sum to 1, as in a transition probability matrix):
        D = C @ (U @ (C.T @ np.ones((C.shape[0], 1))))
        D = np.clip(D, a_min=1, a_max=None)
        D_norm = D ** 0.5
        C = C / D_norm

        # Generate a random space for the kernel projection:
        R = np.random.randn(data.shape[0], projection_dim)

        # Project the approximated kernel on the random space:
        P = C @ (U @ (C.T @ R))

        # Compute the orthonormal representation of the projection array (P):
        Q, _ = np.linalg.qr(P, mode='reduced')

        # Project the approximated kernel on the orthonormal space (Q):
        W = ((Q.T @ C) @ U) @ C.T

        # Execute SVD on the 'compressed' matrix W:
        U, s, Vh = np.linalg.svd(W, full_matrices=False, compute_uv=True, hermitian=False)

        # Project U back to the original space:
        U = Q @ U
        U = U / D_norm

        return U, s, Vh





