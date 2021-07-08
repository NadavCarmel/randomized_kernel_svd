import numpy as np
from scipy import spatial
from typing import List, Tuple
from utils import read_yaml


class KernelApproximation:
    """
    Computes the Nyström approximation of the kernel matrix
    """
    def __init__(self, config_path: str):
        self.configs = read_yaml(yaml_path=config_path)

    @staticmethod
    def calc_exact_kernel(data: np.array, sigma: float) -> np.array:
        # Exact kernel matrix calculation:
        d = spatial.distance.cdist(data, data, metric='sqeuclidean')
        d /= (d.mean() * sigma)
        k = np.exp(-d)
        return 0.5 * (k + k.T)  # only for numerical considerations

    @staticmethod
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

    def run_all(self):
        data_pth = self.configs['data_path']
        data = load_pickle(pickle_path=data_pth)

