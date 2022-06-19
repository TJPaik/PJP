import numpy as np


def trefoil_knot(n=500, noise_level=0.1, std=1.4):
    t = np.random.randn(n) * std

    x = np.sin(t) + 3 * np.sin(2 * t)
    y = np.cos(t) - 3 * np.cos(2 * t)
    z = -np.sin(3 * t)

    return np.vstack((x, y, z)).T + np.random.randn(n, 3) * noise_level


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = trefoil_knot(500, 0, 1.4)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*data.T, alpha=0.5)
    # ax.view_init(80, 10)
    plt.show()
