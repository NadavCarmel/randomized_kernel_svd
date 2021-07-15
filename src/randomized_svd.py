import numpy as np
from fps_sampling import FpsSampling
from kernel_approximation import KernelApproximation
import utils


class RandomizedSVD(FpsSampling, KernelApproximation):

    def run_all(self, data, num_points_to_sample, sigma, projection_dim, m):
        """
        Assume we want to compute the m smallest components for L := I - D**-0.5 @ K @ D**-0.5
        We convert the problem into finding the m largest components of D**-0.5 @ K @ D**-0.5
        Computation is based on all steps of the randomized svd

        :param data: the data (num_samples, point_dim)
        :param num_points_to_sample: number of points to take
        :param sigma: the distance scale hyper-parameter
        :param projection_dim: the new dimension on which we project the kernel
        :param m: the number of SVD components to return
        :return: U, s, Vh -> the svd decomposition of the approximated kernel of order m
        """

        print('start working on randomized_svd')

        # MAIN STEP 1: approximated-kernel building blocks calculation:

        # FPS sampling:
        farthest_idx = self.fps_sampling(point_array=data, num_points_to_sample=num_points_to_sample)

        # Calc C and U:
        C, U = self.calc_C_U(data=data, farthest_idx=farthest_idx, sigma=sigma)

        # Scale the rows of C:
        C, D_norm = self.normalize_C(C, U)

        # MAIN STEP 2: randomized svd:

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
        # U = U / D_norm

        return U, s, Vh

    def run_all(self):
        config_path = '../config.yaml'
        configs = utils.read_yaml(yaml_path=config_path)
        data_pth = configs['data_path']
        data = utils.load_pickle(pickle_path=data_pth)
        num_points_to_sample = configs['num_points_to_sample']
        sigma = configs['sigma']
        projection_dim = configs['projection_dim']
        U, s, Vh = self.randomized_svd(data=data,
                                       num_points_to_sample=num_points_to_sample,
                                       sigma=sigma,
                                       projection_dim=projection_dim)
        return U, s, Vh


if __name__ == '__main__':
    rsvd = RandomizedSVD()
    U, s, Vh = rsvd.run_all()
    print('dine execution')


