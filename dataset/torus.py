import numpy as np


def torus(n=500, noise_level=0.1, a=3, b=1, std1=2, std2=1.5):
    t = np.random.randn(n) * std1
    s = np.random.randn(n) * std2

    x = (b + a * np.cos(t)) * np.cos(s)
    y = (b + a * np.cos(t)) * np.sin(s)
    z = a * np.sin(t)

    data = np.vstack((x, y, z)).T + np.random.randn(n, 3) * noise_level
    return data


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = torus(n=1000, noise_level=0, a=5, b=10, std1=999, std2=999)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*data.T, alpha=0.5)
    ax.view_init(80, 10)
    plt.show()
