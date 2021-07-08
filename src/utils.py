import pickle

import yaml
from datetime import datetime


def read_yaml(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as f:
        yaml_file = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_file


def load_pickle(pickle_path: str):
    with open(pickle_path, 'rb') as f:
        res = pickle.load(f)
    return res


def timeit(my_func):
    def timed(*args, **args_dict):
        s = str(my_func).split(' ')
        print(f'start executing {s[0][1:]}: {s[1]}\n')
        start = datetime.now()
        result = my_func(*args, **args_dict)
        end = datetime.now()
        print(f'*** execution time took: {str(end - start)} *** \n')
        return result
    return timed
