# %%
import matplotlib.pyplot as plt
from cyclonysus.cyclenysus import Cycler
import time
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm


def torus(precision, c, a):
    U = np.linspace(0, 2 * np.pi, precision)
    V = np.linspace(0, 2 * np.pi, precision)
    U, V = np.meshgrid(U, V)
    U = U.flatten()
    V = V.flatten()
    X = (c + a * np.cos(V)) * np.cos(U)
    Y = (c + a * np.cos(V)) * np.sin(U)
    Z = a * np.sin(V)
    return X, Y, Z


# %%
def torus_random(n_points, c, a):
    U = np.random.uniform(0, 2 * np.pi, n_points)
    V = np.random.uniform(0, 2 * np.pi, n_points)
    X = (c + a * np.cos(V)) * np.cos(U)
    Y = (c + a * np.cos(V)) * np.sin(U)
    Z = a * np.sin(V)
    return X, Y, Z


# x, y, z = torus_random(100, 3, 1)
x, y, z = torus_random(10000, 3, 1)
X = np.stack([x, y, z], 1) * 100
# X = np.concatenate(
#     [np.random.rand(300, 3), np.random.rand(300, 3) + 2, np.random.rand(300, 3) - 5, np.random.rand(300, 3) + 20])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*X.T, s=1)
ax.view_init(elev=80., azim=0)
plt.show()
# %%

X_torch = torch.from_numpy(X).float()
ds = TensorDataset(X_torch, )
dl = DataLoader(ds, 512, True)

km = KMeans(8)
km.fit(X)
cluster_centers = km.cluster_centers_

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*X.T, s=1, c=km.predict(X))
ax.scatter(*cluster_centers.T, c='red', s=100)
# ax.scatter(*X_torch.T, s=1, alpha=0.1)
ax.view_init(elev=60, azim=0)
plt.show()


# %%
class LastModel(nn.Module):
    def __init__(self, cluster_centers):
        super(LastModel, self).__init__()
        self.cluster_centers = nn.Parameter(torch.from_numpy(cluster_centers).float())

    def forward(self, x):
        # batch x
        encoding = x

        distances1 = torch.cdist(encoding, self.cluster_centers, p=2.)
        distances2 = (distances1 ** 2) + 1
        distances3 = 1 / distances2
        q_ij = distances3 / distances3.sum(1)[:, None]
        f_j = q_ij.sum(0)
        p_ij = (q_ij ** 2) / f_j[None, :]
        p_ij = p_ij / p_ij.sum(1)[:, None]

        return p_ij, q_ij, f_j


last_model = LastModel(cluster_centers)
optimizer = optim.Adam(last_model.parameters(), lr=0.005)
loss_fn = nn.KLDivLoss(reduction='batchmean')

pbar = tqdm(range(3000))
cycler = Cycler()
for epoch in pbar:
    for batch, in dl:
        last_model.train()
        last_model.train()

        p_ij, q_ij, f_j = last_model(batch)
        loss1 = loss_fn(q_ij.log(), p_ij)

        loss2 = torch.tensor(0)
        for cluster_no in range(8):
            tmp_data = batch[
                q_ij.argmax(1) == cluster_no
                ].numpy()
            if len(tmp_data) == 0:
                continue
            cycler.fit(tmp_data)

            cycle = cycler.longest_intervals(1)
            if len(cycle) == 0:
                continue
            cycle = cycle[0]
            cycle_index = np.unique(cycler.get_cycle(cycle).flatten())
            barcode = cycler.barcode
            max_persistence = np.max(barcode[:, 1] - barcode[:, 0])

            loss2 = loss2 + torch.std(q_ij[cycle_index, cluster_no]) * max_persistence

        loss = loss1 * 0 + loss2 * 1000

        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        pbar.set_postfix({'loss1': loss1.item(), 'loss2': loss2.item()})
# %%
with torch.no_grad():
    p_ij, q_ij, f_j = last_model(X_torch)
# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*X.T, s=1, c=q_ij.argmax(1).detach().cpu().numpy())
ax.scatter(*last_model.cluster_centers.detach().cpu().numpy().T, c='red', s=100)
# ax.scatter(*X_torch.T, s=1, alpha=0.1)
ax.view_init(elev=60, azim=0)
plt.show()
# %%
