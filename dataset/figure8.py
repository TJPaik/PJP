import numpy as np
import matplotlib.pyplot as plt
from .circle import circle


def figure8(n=200, noise_level=0.05):
    data = np.vstack([
        circle(100, 0.0, 1.5, _param=1),
        (circle(100, 0, 1.5, _param=2) + [3, 0]) * 0.5
    ]) + np.random.randn(n, 2) * noise_level
    return data


if __name__ == '__main__':
    data = figure8()

    plt.scatter(*data.T)
    plt.show()
