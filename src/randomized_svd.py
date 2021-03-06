from typing import Tuple
import numpy as np
from fps_sampling import FpsSampling
from kernel_approximation import KernelApproximation
import utils


class RandomizedSVD(FpsSampling, KernelApproximation):

    @staticmethod
    def randomized_svd(C_scaled: np.array, U_inv: np.array, projection_dim: int, power_iter: int = 30) -> Tuple[np.array, np.array, np.array]:
        """
        Assume we want to compute the 'projection_dim' smallest components for L := I - D**-0.5 @ K @ D**-0.5
        We convert the problem into finding the 'projection_dim' largest components of D**-0.5 @ K @ D**-0.5
        Computation is based on all steps of the randomized svd

        :param C_scaled: the scaled (row-wise) sampled columns
        :param U_inv: the sampled rows + columns
        :param projection_dim: the new dimension on which we project the kernel (also, the number of SVD components returned)
        :return: U, s, Vh -> the SVD decomposition of the approximated kernel
        """

        print('start working on randomized_svd')

        # Generate a random space for the kernel projection:
        R = np.random.randn(C_scaled.shape[0], projection_dim)

        # Project the approximated kernel on the random space:
        P = C_scaled @ (U_inv @ (C_scaled.T @ R))

        # Compute the orthonormal representation of the projection array (P):
        Q, _ = np.linalg.qr(P, mode='reduced')

        # Project the approximated kernel on the orthonormal space (Q):
        W = ((Q.T @ C_scaled) @ U_inv) @ C_scaled.T

        # Execute SVD on the 'compressed' matrix W:
        U, s, Vh = np.linalg.svd(W, full_matrices=False, compute_uv=True, hermitian=False)

        # Project U back to the original space:
        U = Q @ U

        return U, s, Vh

    def run_all(self, config_path):
        configs = utils.read_yaml(yaml_path=config_path)
        data_path = configs['data_path']
        data = utils.load_pickle(pickle_path=data_path)
        num_points_to_sample = configs['num_points_to_sample']
        sigma = configs['sigma']
        projection_dim = configs['projection_dim']

        # MAIN STEP 1: approximated-kernel building blocks calculation:

        # FPS sampling:
        farthest_idx = self.fps_sampling(point_array=data,
                                         num_points_to_sample=num_points_to_sample)

        # Calc C and U:
        C, U_inv = self.calc_C_U(data=data,
                                 farthest_idx=farthest_idx,
                                 sigma=sigma)

        # Scale the rows of C:
        C_scaled = self.normalize_C(C=C, U_inv=U_inv)

        # MAIN STEP 2: execute the randomized SVD flow:

        U, s, Vh = self.randomized_svd(C_scaled=C_scaled,
                                       U_inv=U_inv,
                                       projection_dim=projection_dim)

        return U, s, Vh


if __name__ == '__main__':
    config_path = '../config.yaml'
    rsvd = RandomizedSVD()
    U, s, Vh = rsvd.run_all(config_path=config_path)

    # compare with the exact SVD spectrum:
    from error_analysis import calc_spectral_ratio
    calc_spectral_ratio(s_approx=s, config_path=config_path)

    print('done execution')





