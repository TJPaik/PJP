# %%
# import libraries
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from cyclonysus.cyclenysus import Cycler

#%%
# create dataset
def make_three_circles(n_points=1024):
    centers = [(0,0), (2.5,0), (1.25,2.25)]
    circles = []

    for center in centers:
        theta = np.random.random(n_points)
        x = np.cos(2*np.pi*theta) + center[0]
        y = np.sin(2*np.pi*theta) + center[1]
        circle = np.stack((x, y), axis=1)
        circles.append(circle)
    return np.vstack(circles)

X = make_three_circles()
plt.figure(figsize=(4,4))
plt.scatter(X[:,0], X[:,1], s=1)
plt.show()

#%%
# k-means clustering
n_clusters = 3
km = KMeans(n_clusters)
km.fit(X)
cluster_centers = km.cluster_centers_

plt.figure(figsize=(5,4))
plt.scatter(X[:,0], X[:,1], s=1, c=km.labels_)
plt.colorbar()
plt.show()

# %%
# define soft clustering model
class LastModel(nn.Module):
    '''
    Model for soft clustering on latent space as defined in DEC paper.
        cluster_centers: 2d array of shape (n_clusters,latent_dim)
    '''
    def __init__(self, cluster_centers):
        super(LastModel, self).__init__()
        self.cluster_centers = nn.Parameter(torch.from_numpy(cluster_centers).float())

    def forward(self, x):
        '''
        Calcutate soft clustering probabilities
            x: batch having shape (batch_size,latent_dim)
        '''
        encoding = x # latent embedding (currently set as identity map)
        '''
            distances1: sqrt(|z_i - mu_j|^2)
            distances2: 1 + |z_i - mu_j|^2
            distances3: (1 + |z_i - mu_j|^2)^(-1)
            all have shape (batch_size,n_clusters)
        '''
        distances1 = torch.cdist(encoding, self.cluster_centers, p=2.) # 
        distances2 = (distances1 ** 2) + 1
        distances3 = 1 / distances2

        q_ij = distances3 / distances3.sum(1)[:, None]
        f_j = q_ij.sum(0)
        p_ij = (q_ij ** 2) / f_j[None, :]
        p_ij = p_ij / p_ij.sum(1)[:, None]

        return p_ij, q_ij, f_j

# %%
# hyperparameters
batch_size = 256
lr = 0.005

# torch dataloader
X_torch = torch.from_numpy(X).float()
ds = TensorDataset(X_torch, )
dl = DataLoader(ds, batch_size, shuffle=True)

#%%
# initialize model
last_model = LastModel(cluster_centers)
optimizer = optim.Adam(last_model.parameters(), lr=lr)
loss_fn = nn.KLDivLoss(reduction='batchmean')
cycler = Cycler()

#%%
# train model
pbar = tqdm(range(500))

for epoch in pbar:
    for batch_idx,(batch,) in enumerate(dl):
        '''
            batch: 2d tensor of shape (batch_size,latent_dim)
        '''
        # initialize figure
        is_last_batch = (batch_idx == len(dl)-1)
        if is_last_batch:
            plt.clf()
            plt.figure(figsize=(5,4))

        # start training
        last_model.train()

        # DEC loss
        p_ij, q_ij, f_j = last_model(batch)
        loss_DEC = loss_fn(q_ij.log(), p_ij)

        # Cycle Loss
        loss_cycle = torch.tensor(0) # initialize loss
        for cluster_no in range(n_clusters): # loop through clusters
            # select cluster
            cluster = batch[q_ij.argmax(1) == cluster_no].numpy() # current cluster
            if len(cluster) == 0:
                continue
            
            # find most persistent cycle
            cycler.fit(cluster)
            longest_barcodes = cycler.longest_intervals(n=1) # returns list of top n longest barcodes
            if len(longest_barcodes) == 0:
                continue
            barcode = longest_barcodes[0]
            cycle = cycler.get_cycle(barcode) # e.g. [[1 4][2 3][2 4][1 3]]
            cycle_vertices = cycler.order_vertices(cycle) # e.g. [1 4 2 3]

            # calculate max persistence
            barcodes = cycler.barcode
            max_persistence = np.max(barcodes[:, 1] - barcodes[:, 0])

            # update loss
            prob_std = torch.std(q_ij[cycle_vertices, cluster_no])
            loss_cycle = loss_cycle - prob_std * max_persistence

            # plot cycle if last batch
            if is_last_batch:
                plt.plot(cluster[cycle_vertices][:,0], 
                        cluster[cycle_vertices][:,1], 
                        lw=3,
                        label=f'std {prob_std:.2f}, pers {max_persistence:.2f}')

        loss = loss_DEC * 0 + loss_cycle * 1000

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'loss1': loss_DEC.item(), 'loss2': loss_cycle.item()})

    # plot data points
    plt.title(f'EPOCH {epoch}\nLOSS {loss_cycle:.3f}\n\n', loc='left')
    plt.scatter(batch[:,0], batch[:,1], c=q_ij.argmax(1).detach().cpu().numpy())
    centers = last_model.cluster_centers.detach().cpu().numpy()
    plt.scatter(centers[:,0], centers[:,1], s=10, c='red')
    plt.legend(bbox_to_anchor=(1,1.3))
    # plt.savefig(f'./pics/0510/pics1/circles_{epoch}', bbox_inches='tight')
    # plt.show()
    plt.close()
