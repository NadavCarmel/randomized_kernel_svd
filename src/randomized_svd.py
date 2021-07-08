import numpy as np
from fps_sampling import FpsSampling
from kernel_approximation import KernelApproximation
import utils


class RandomizedSVD(FpsSampling, KernelApproximation):

    def randomized_svd(self, data, n_sampling_points, sigma, projection_dim):

        # FPS sampling:
        farthest_idx = self.fps_sampling(point_array=data, num_points_to_sample=n_sampling_points)

        # Calc C and U:
        C, U = self.calc_C_U(data=data, farthest_idx=farthest_idx, sigma=sigma)

        # Generate a random space for the kernel projection:
        R = np.random.randn(data.shape[0], projection_dim)

        # Project the approximated kernel on the random space:
        P = C @ (U @ (C.T @ R))

        # Compute the orthonormal representation of the projection array (P):
        Q, _ = np.linalg.qr(P, mode='reduced')

        # Project the approximated kernel on the orthonormal space (Q):



