import numpy as np
from scipy import spatial
from typing import List, Tuple


class KernelApproximation:
    """
    Computes Nyström approximation of a kernel matrix.
    """

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
        A Nyström approximation of the kernel matrix.
        Construct sub-matrices C & U (no need for R - the kernel matrix is symmetric -> R = C.T):
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
        U = np.linalg.pinv(U)
        print(f'C.shape = {C.shape}')
        print(f'U.shape = {U.shape}')
        return C, U


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
    n_sampling_points = configs['n_sampling_points']
    fs = FpsSampling()
    farthest_idx = fs.fps_sampling(point_array=data, num_points_to_sample=n_sampling_points)
    # calc kernel approximation C and U:
    sigma = configs['sigma']
    ka = KernelApproximation()
    C, U = ka.calc_C_U(data=data, farthest_idx=farthest_idx, sigma=sigma)
    print('done execution')
