import numpy as np
from scipy import spatial
from typing import List, Tuple


class KernelApproximation:

    @staticmethod
    def calc_exact_kernel(data: np.array, sigma: float) -> np.array:
        # Exact kernel matrix calculation:
        print('calculate exact kernel')
        d = spatial.distance.cdist(data, data, metric='sqeuclidean')
        d /= (d.mean() * sigma)
        k = np.exp(-d)
        return 0.5 * (k + k.T)  # only for numerical considerations

    @staticmethod
    def calc_C_U(data: np.array, farthest_idx: List[int], sigma: float) -> Tuple[np.array, np.array]:
        """
        Compute the Nyström approximation matrices of the kernel matrix (so that K ~ C @ U @ C.T).
        :param data: the data (num_samples, point_dim)
        :param farthest_idx: the indices of the sampled points
        :param sigma: the distance scale hyper-parameter
        :return: C -> a subset of columns of the original kernel, U -> a subset of rows of C.
        """
        print('start working on calc_C_U')
        d = spatial.distance.cdist(data, data[farthest_idx], metric='sqeuclidean')
        d /= sigma
        C = np.exp(-d)
        U = C[farthest_idx, :]
        U_inv = np.linalg.pinv(U)
        return C, U_inv

    @staticmethod
    def normalize_C(C: np.array, U_inv: np.array) -> Tuple[np.array, np.array]:
        print('start working on normalize_C')
        # Scale the rows of C:
        D = C @ (U_inv @ (C.T @ np.ones((C.shape[0], 1))))  # efficient summation of each row in the approximated kernel
        D = np.clip(D, a_min=1, a_max=None)  # clip D so that each element is at least 1, as in the sum of the original kernel matrix
        D_sqrt = D ** 0.5
        C_scaled = C / D_sqrt  # this will convert K to D ** -0.5 @ K @ D ** -0.5
        return C_scaled, D_sqrt


if __name__ == '__main__':
    import utils
    from fps_sampling import FpsSampling
    # load config
    config_path = '../config.yaml'
    configs = utils.read_yaml(yaml_path=config_path)
    # load data:
    data_path = configs['data_path']
    data = utils.load_pickle(pickle_path=data_path)
    # calc farthest-points-idx:
    num_points_to_sample = configs['num_points_to_sample']
    fs = FpsSampling()
    farthest_idx = fs.fps_sampling(point_array=data, num_points_to_sample=num_points_to_sample)
    # calc kernel approximation C and U:
    sigma = configs['sigma']
    ka = KernelApproximation()
    C, U_inv = ka.calc_C_U(data=data, farthest_idx=farthest_idx, sigma=sigma)
    C_scaled, D_norm = ka.normalize_C(C=C, U_inv=U_inv)
    print('done execution')
