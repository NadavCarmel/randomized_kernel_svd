import numpy as np
from scipy import spatial
from typing import List, Tuple
from utils import read_yaml, load_pickle, timeit
from fps_sampling import FpsSampling


class KernelApproximation:
    """
    Computes Nyström approximation of a kernel matrix.
    """

    @staticmethod
    def calc_exact_kernel(data: np.array, sigma: float) -> np.array:
        # Exact kernel matrix calculation:
        d = spatial.distance.cdist(data, data, metric='sqeuclidean')
        d /= (d.mean() * sigma)
        k = np.exp(-d)
        return 0.5 * (k + k.T)  # only for numerical considerations

    @staticmethod
    @timeit
    def calc_C_U(data: np.array, farthest_idx: List[int], sigma: float) -> Tuple[np.array, np.array]:
        # A Nyström approximation of the kernel matrix.
        # Construct sub-matrices C & U (no need for R - the kernel matrix is symmetric -> R = C.T):
        d = spatial.distance.cdist(data, data[farthest_idx], metric='sqeuclidean')
        d /= sigma
        C = np.exp(-d)
        U = C[farthest_idx, :]
        U = np.linalg.pinv(U)
        print(f'C.shape = {C.shape}')
        print(f'U.shape = {U.shape}')
        return C, U


if __name__ == '__main__':
    # load config
    config_path = '../config.yaml'
    configs = read_yaml(yaml_path=config_path)
    # load data:
    data_pth = configs['data_path']
    data = load_pickle(pickle_path=data_pth)
    # calc farthest-points-idx:
    n_sampling_points = configs['n_sampling_points']
    fs = FpsSampling()
    farthest_idx = fs.fps_sampling(point_array=data, num_points_to_sample=n_sampling_points)
    # calc kernel approximation C and U:
    sigma = configs['sigma']
    ka = KernelApproximation()
    C, U = ka.calc_C_U(data=data, farthest_idx=farthest_idx, sigma=sigma)
    print('done execution')
