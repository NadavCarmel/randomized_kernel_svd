import numpy as np
import utils
from kernel_approximation import KernelApproximation


def calc_spectral_ratio(s_approx, config_path):
    """
    calc exact SVD (for error analysis), and compare with the approximated result
    :param s_approx: the singular values of the approximated method
    :param config_path: config path
    """

    configs = utils.read_yaml(yaml_path=config_path)
    sigma = configs['sigma']
    data_path = configs['data_path']
    data = utils.load_pickle(pickle_path=data_path)

    # construct the normalized kernel:
    K_exact = KernelApproximation.calc_exact_kernel(data=data, sigma=sigma)
    D_exact = np.sum(K_exact, axis=1)
    D_exact_sqrt = np.diag(D_exact ** 0.5)
    D_exact_sqrt_inv = np.linalg.inv(D_exact_sqrt)
    K_exact_normed = D_exact_sqrt_inv @ K_exact @ D_exact_sqrt_inv

    # exact SVD compute:
    U_exact, s_exact, Vh_exact = np.linalg.svd(K_exact_normed, full_matrices=False, compute_uv=True, hermitian=True)

    print(s_approx[:10] / s_exact[:10])