# %%
import matplotlib.pyplot as plt
import sklearn
import torch

import circularcoordinates
import kmapper as km
from kmapper.circular_cover import CircCover

# %%
N = 350
data = torch.randn(N, 2)
data = data / torch.linalg.norm(data, dim=1).view(-1, 1) + torch.randn(N, 2) * 0.1

data = data.numpy()
plt.scatter(*data.T)
plt.show()
# %%
mapper = km.KeplerMapper(verbose=2)

circ = circularcoordinates.circular_coordinate(prime=11)
lens = circ.fit_transform(data, weight=False).reshape(-1, 1)
# %%
plt.scatter(*data.T, c=lens[:, 0], cmap='hsv')
plt.show()
# %%
graph = mapper.map(
    lens=lens,
    X=data,
    clusterer=sklearn.cluster.DBSCAN(eps=0.2, min_samples=5),
    cover=CircCover(n_cubes=8, perc_overlap=0.2),
)

km.draw_matplotlib(graph)

plt.show()
# %%
