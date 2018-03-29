import os

import numpy as np
from torch.utils import data

EPS = 1e-9
FIELDS = [
    'x0',
    'x1',
    'x2',
    'y0',
    'y1',
    'y2',
    'x_0',
    'x_1',
    'x_2',
    'y_0',
    'y_1',
    'y_2',
    'u0',
    'u1',
    'u2',
    'u3',
    'u4',
    'w',
    'h',
    'm',
    'I',
    'µ',
    'type'
]

# x = row[['y0', 'y1', 'y2', 'w', 'h', 'm', 'µ']].values
# u = row[['u0', 'u1', 'u2', 'u3']].values
# y_ = row[['y_0', 'y_1', 'y_2']].values


def file_iterator(filename):
    while True:
        with open(filename, 'r') as f:
            for line in f:
                floats = list(map(np.float32, line.split(',')[:-1]))
                v = dict(zip(FIELDS[:-1], floats))
                x = np.array([v['y0'], v['y1'], v['y2'], v['w'], v['h'], v['m'], v['µ']])
                u = np.array([v['u0'], v['u1'], v['u2'], v['u3']])
                y = np.zeros(4).astype(np.float32)
                y[0] = v['y_0']
                y[1] = v['y_1']
                y[2] = np.cos(v['y_2'])
                y[3] = np.sin(v['y_2'])
                yd = y[:2] - x[:2]
                if np.linalg.norm(yd) < EPS and np.random.rand() < 0.95:
                    continue
                x[:3] += np.random.randn(3) * np.array([0.02, 0.02, 0.0])
                x[-2] *= np.random.uniform(0.7, 1.3) # mass
                x[-1] *= np.random.uniform(0.7, 1.3) # friction
                yield x, u, y


def files_iterator(paths):
    iterators = [file_iterator(path) for path in paths]
    while True:
        iterator = np.random.choice(iterators)
        yield next(iterator)


class PushingDataset(data.Dataset):

    def __init__(self, folder_path):
        np.random.seed
        files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
        self.iterator = files_iterator(files)

    def __getitem__(self, i):
        return next(self.iterator)

    def __len__(self):
        return 10000000

    def worker_init(self, worker_id):
        np.random.seed(worker_id)



if __name__ == '__main__':
    dataset = PushingDataset('training_data/dual_point_one_size_2/train')
    for i in range(10):
        x, u, y = dataset[2]
        print(x[:3], y)
