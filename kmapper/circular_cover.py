import numpy as np

from kmapper.cover import Cover


class CircCover(Cover):
    def __init__(self, n_cubes=10, perc_overlap=0.5):
        super(CircCover, self).__init__(n_cubes=n_cubes, perc_overlap=perc_overlap, limits=None, verbose=0)
        self.mine = 1 / (2 * (n_cubes * (1 - perc_overlap)))

    def fit(self, lens):
        self.centers_ = np.asarray([[0], *[[i / self.n_cubes] for i in range(1, self.n_cubes)]])
        return self.centers_

    @staticmethod
    def diff(a, b):
        p = a - b
        if p > 0:
            return p if p < 0.5 else 1 - p
        else:
            return -p if -p < 0.5 else 1 + p

    def transform_single(self, lens, center, i=0):
        assert lens.shape[1] == 2
        assert center.shape == (1,)

        diff = [self.diff(el, center) for el in lens[:, 1]]
        in_radius = [True if el < self.mine else False for el in diff]
        hypercube = lens[in_radius]

        return hypercube

    def transform(self, lens, centers=None):
        centers = centers or self.centers_
        hypercubes = [
            self.transform_single(lens, cube, i) for i, cube in enumerate(centers)
        ]
        hypercubes = [cube for cube in hypercubes if len(cube)]
        return hypercubes
