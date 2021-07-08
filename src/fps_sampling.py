import numpy as np
from typing import List


class FpsSampling:
    """
    Compute farthest-point-sampling given a data array and number of points to sample.
    """

    @staticmethod
    def calc_distances(p0: np.array, point_array: np.array) -> float:
        """
        This method calculates the distance between one point and the rest.
        :param p0: The point to measure against the rest
        :param point_array: Array of all points
        :return: numpy array (vector) of distance from p_0 to the point at respective point's idx (at that idx)
        """
        return ((p0 - point_array) ** 2).sum(axis=1)

    def fps_sampling(self, point_array: np.array, num_points_to_sample: int) -> List[int]:
        """
        This method performs farthest-point-sampling.
        :param point_array: the data (num_samples, point_dim)
        :param num_points_to_sample: number of points to take
        :return: farthest_idx -> the indices of the sampled points
        """

        print('start working on fps_sampling')

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
            farthest_id = np.argmax(distances)
            farthest_idx.append(farthest_id)
            farthest_points[i] = point_array[farthest_id]
            # Look at minimum between:
            # 1. distance from current point
            # 2. minimal distance from all sampled points so far
            distances = np.minimum(distances, self.calc_distances(farthest_points[i], point_array))
            i += 1

        return farthest_idx


if __name__ == '__main__':
    import utils
    config_path = '../config.yaml'
    configs = utils.read_yaml(yaml_path=config_path)
    data_pth = configs['data_path']
    data = utils.load_pickle(pickle_path=data_pth)
    num_points_to_sample = configs['num_points_to_sample']
    fs = FpsSampling()
    farthest_idx = fs.fps_sampling(point_array=data, num_points_to_sample=num_points_to_sample)
    print('done execution')


