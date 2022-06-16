# %%
import circularcoordinates
from ripser import ripser
import numpy as np

import matplotlib.pyplot as plt

# %%
N = 100
data = np.random.random((N, 2)) - (0.25, 0.25)
data /= np.linalg.norm(data, axis=1)[:, None]
data += np.random.random((N, 2)) * 0.1

data1 = np.random.random((N, 2)) - (0.25, 0.25)
data1 /= np.linalg.norm(data1, axis=1)[:, None]
data1 += np.random.random((N, 2)) * 0.1

data2 = np.random.random((N, 2)) - (0.3, 0.3)
data2 /= np.linalg.norm(data2, axis=1)[:, None]
data2 += np.random.random((N, 2)) * 0.1

data = np.concatenate([data, data1 + (2, 3.5), data2 + (4, 0)])

plt.scatter(*data.T)
plt.show()
# %%
result = ripser(data, coeff=2, do_cocycles=True)
diagrams = result['dgms']
cocycles = result['cocycles']
D = result['dperm2all']
dgm1 = diagrams[1]
idx = np.argmax(dgm1[:, 1] - dgm1[:, 0])
# %%
circ = circularcoordinates.circular_coordinate(prime=11)
vertex_value1 = circ.fit_transform(data, weight=True)
circ.plot_pca(data, vertex_values=vertex_value1)
