import numpy as np
from fps_sampling import FpsSampling
from kernel_approximation import KernelApproximation
import utils


class RandomizedSVD(FpsSampling, KernelApproximation):

    def randomized_svd(self, C_scaled: np.array, D_sqrt: np.array, U: np.array, projection_dim: int):
        """
        Assume we want to compute the m smallest components for L := I - D**-0.5 @ K @ D**-0.5
        We convert the problem into finding the m largest components of D**-0.5 @ K @ D**-0.5
        Computation is based on all steps of the randomized svd

        :param C_scaled: the scaled (row-wise) sampled columns
        :param D_norm: the normalizer of C (sum of each row of C ** 0.5)
        :param U: the sampled rows + columns
        :param projection_dim: the new dimension on which we project the kernel (also, the number of SVD components returned)
        :return: U, s, Vh -> the (truncated) SVD decomposition of the approximated kernel
        """

        print('start working on randomized_svd')

        # Generate a random space for the kernel projection:
        R = np.random.randn(C_scaled.shape[0], projection_dim)

        # Project the approximated kernel on the random space:
        P = C_scaled @ (U @ (C_scaled.T @ R))

        # Compute the orthonormal representation of the projection array (P):
        Q, _ = np.linalg.qr(P, mode='reduced')

        # Project the approximated kernel on the orthonormal space (Q):
        W = ((Q.T @ C_scaled) @ U) @ C_scaled.T

        # Execute SVD on the 'compressed' matrix W:
        U, s, Vh = np.linalg.svd(W, full_matrices=False, compute_uv=True, hermitian=False)

        # Project U back to the original space:
        U = Q @ U

        return U, s, Vh

    def run_all(self, config_path):
        configs = utils.read_yaml(yaml_path=config_path)
        data_pth = configs['data_path']
        data = utils.load_pickle(pickle_path=data_pth)
        data = data[:10, :]  # todo: remove
        num_points_to_sample = configs['num_points_to_sample']
        sigma = configs['sigma']
        projection_dim = configs['projection_dim']

        # MAIN STEP 1: approximated-kernel building blocks calculation:

        # FPS sampling:
        farthest_idx = self.fps_sampling(point_array=data,
                                         num_points_to_sample=num_points_to_sample)

        farthest_idx.sort()  # todo: remove

        # Calc C and U:
        C, U = self.calc_C_U(data=data,
                             farthest_idx=farthest_idx,
                             sigma=sigma)

        # Scale the rows of C:
        C_scaled, D_sqrt = self.normalize_C(C=C, U=U)

        # MAIN STEP 2: execute the randomized SVD flow:

        U, s, Vh = self.randomized_svd(C_scaled=C_scaled,
                                       D_sqrt=D_sqrt,
                                       U=U,
                                       projection_dim=projection_dim)

        return U, s, Vh


if __name__ == '__main__':
    config_path = '../config.yaml'
    rsvd = RandomizedSVD()
    U, s, Vh = rsvd.run_all(config_path=config_path)
    print('dine execution')


# TESTS:

# from scipy.spatial.distance import cdist
# d = cdist(data[farthest_idx],data[farthest_idx], metric='sqeuclidean')
# K = np.exp(-d/sigma)

# test Symmetric normalized Laplacian:
# D_sqrt_diag = np.diag(np.squeeze(D_sqrt))
# K_norm = np.linalg.inv(D_sqrt_diag) @ K @ np.linalg.inv(D_sqrt_diag)
# U_test, s_test, Vh_test = np.linalg.svd(K_norm, full_matrices=False, compute_uv=True, hermitian=False)

# test Random walk normalized Laplacian:
# U = U * D_sqrt  # if we want to compute the random walk Laplacian SVD (L := I - D ** -1 @ K) (**not correct mathematically**)

# D_diag = np.diag(np.squeeze(D_sqrt **  2))
# K_norm = np.linalg.inv(D_diag) @ K
# U_test, s_test, Vh_test = np.linalg.svd(K_norm, full_matrices=False, compute_uv=True, hermitian=False)
# [e_k, v_k] = np.linalg.eigh(K_norm @ K_norm.T)


