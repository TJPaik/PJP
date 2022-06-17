# %%
import torch
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from kmapper.circular_cover import CircCover
import kmapper as km
import dionysus as d
from torch import nn, optim
import matplotlib.pyplot as plt

# %%
N = 350
data = torch.randn(N, 2)
data = data / torch.linalg.norm(data, dim=1).view(-1, 1) + torch.randn(N, 2) * 0.1

# data = torch.cat([data, torch.randn(N, 2) + 20])

data = data.numpy()
plt.scatter(*data.T)
plt.show()
# %%
mapper = km.KeplerMapper(verbose=2)
lens = np.angle([complex(*el) for el in data]).reshape(-1, 1) / (2 * np.pi) + 0.5
# %%
plt.scatter(*data.T, c=lens[:, 0], cmap='hsv')
plt.show()
# %%
graph = mapper.map(
    lens=lens,
    X=data,
    clusterer=sklearn.cluster.DBSCAN(eps=0.2, min_samples=1),
    cover=CircCover(n_cubes=8, perc_overlap=0.2),
)

km.draw_matplotlib(graph)

plt.show()
# %%
