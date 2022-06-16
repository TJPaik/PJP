# %%
import dionysus as d
import numpy as np
import matplotlib.pyplot as plt

# %%
simplices = [
    ([0], 0),
    ([1], 0),
    ([2], 0),
    ([3], 0),
    ([0, 1], 0),
    ([1, 2], 0),
    ([2, 3], 0),
    ([0, 3], 0),
]
# %%
f = d.Filtration()
for vertices, time in simplices:
    f.append(d.Simplex(vertices, time))
# %%
prime = 2
p = d.cohomology_persistence(f, prime, True)
dgms = d.init_diagrams(p, f)
# %%
pt = max(dgms[1], key=lambda pt: pt.death - pt.birth)
print(pt)
# %%
cocycle = p.cocycle(pt.data)
print(cocycle)
# %%
f_restricted = d.Filtration([s for s in f if s.data <= (pt.death + pt.birth) / 2])
vertex_values = d.smooth(f_restricted, cocycle, prime)
# %%
print(vertex_values)
# %%
points = np.random.normal(size=(50, 2))
for i in range(points.shape[0]):
    points[i] = points[i] / np.linalg.norm(points[i], ord=2) * np.random.uniform(1, 1.5)

plt.scatter(*points.T)
plt.show()
#%%

prime = 11
f = d.fill_rips(points, 2, 2.)
p = d.cohomology_persistence(f, prime, True)
dgms = d.init_diagrams(p, f)

d.plot.plot_bars(dgms[1], show=True)

pt = max(dgms[1], key=lambda pt: pt.death - pt.birth)
print(pt)

cocycle = p.cocycle(pt.data)
# %%
f_restricted = d.Filtration([s for s in f if s.data <= (pt.death + pt.birth) / 2])
vertex_values = d.smooth(f_restricted, cocycle, prime)
# %%
plt.scatter(points[:, 0], points[:, 1], c=vertex_values, cmap='hsv')
plt.show()
