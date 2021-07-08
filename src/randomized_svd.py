from fps_sampling import FpsSampling
from kernel_approximation import KernelApproximation


class RandomizedSVD(FpsSampling, KernelApproximation):

    def randomized_svd(self):

        # FPS sampling:
        farthest_idx = self.fps_sampling(X, num_points_to_sample=self.n_FPS)

        # Calc C and U:
        C, U = self.calc_C_U(X, farthest_idx)




