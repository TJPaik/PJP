import numpy as np


def circle(n=500, noise_level=0.2, std=3, _param=0):
    t = np.random.randn(n) * std + _param
    t.sort()
    x = np.cos(t)
    y = np.sin(t)

    return np.vstack((x, y)).T + np.random.randn(n, 2) * noise_level


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = circle(500, 0, 1)

    plt.scatter(*data.T, alpha=0.5)
    plt.show()
