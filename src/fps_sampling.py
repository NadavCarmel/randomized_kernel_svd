import numpy as np
from utils import read_yaml, load_pickle, timeit


class FpsSampling:
    """
    Computes the farthest-point-sampling of a dataset X
    """
    def __init__(self, config_path: str):
        self.configs = read_yaml(yaml_path=config_path)

    @staticmethod
    def calc_distances(p0: np.array, point_array: np.array) -> float:
        """
        This method calculates the distance between one point and the rest.
        :param p0: The point to measure against the rest
        :param point_array: Array of all points
        :return: numpy array (vector) of distance from p_0 to the point at respective point's idx (at that idx)
        """
        return ((p0 - point_array) ** 2).sum(axis=1)

    @timeit
    def fps_sampling(self, point_array: np.array, num_points_to_sample: int):
        """
        This function performs farthest point sampling.
        :param point_array: the data (num_samples, point_dim)
        :param num_points_to_sample: number of points to take
        :return: the index of the sampled points
        """

        if num_points_to_sample is None or num_points_to_sample > point_array.shape[0]:
            num_points_to_sample = point_array.shape[0]

        num_points, point_dim = point_array.shape
        farthest_points = np.zeros((num_points_to_sample, point_dim))
        farthest_idx = []
        # Randomly pick first point
        np.random.seed(1)
        first_idx = np.random.randint(num_points)
        farthest_idx.append(first_idx)
        farthest_points[0] = point_array[first_idx]

        # Calculate distances from starting point
        distances = self.calc_distances(farthest_points[0], point_array)
        i = 1
        while i < num_points_to_sample:  # and tmp_dist > min_distance_threshold
            # Pick farthest point:
            des_idx = np.argmax(distances)
            farthest_idx.append(des_idx)
            farthest_points[i] = point_array[des_idx]
            # Look at minimum between:
            # 1. distance from current point
            # 2. minimal distance from all sampled points so far
            distances = np.minimum(distances, self.calc_distances(farthest_points[i], point_array))
            i += 1

        print('done sampling')
        return farthest_idx

    def run_all(self):
        data_pth = self.configs['data_path']
        data = load_pickle(pickle_path=data_pth)
        farthest_idx = self.fps_sampling(point_array=data, num_points_to_sample=10)
        return farthest_idx


if __name__ == '__main__':
    config_path = '../config.yaml'
    fs = FpsSampling(config_path=config_path)
    fs.run_all()
    print('done')

