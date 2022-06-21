from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.online_kmeans import OnlineKMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import geomstats.backend as gs
import geomstats.visualization as visualization
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

sphere = Hypersphere(dim=5)

data = sphere.random_uniform(n_samples=10)

clustering = OnlineKMeans(metric=sphere.metric, n_clusters=4)
clustering = clustering.fit(data)

import numpy as np
x = np.array([np.random.random() for _ in range(2000)])
y = np.array([np.random.random() for _ in range(2000)])
z = np.array([np.random.random() for _ in range(2000)])

norm_x = np.sqrt(np.sum(x**2))
norm_y = np.sqrt(np.sum(y**2))
norm_z = np.sqrt(np.sum(z**2))

x = x/norm_x
y = y/norm_y
z = z/norm_z

norm = np.sqrt(np.sum(x**2))
norm
x = []
y = []
z = []
for i in range(200):
    u = np.random.normal(0,1)
    v = np.random.normal(0,1)
    w = np.random.normal(0,1)
    norm = (u*u + v*v + w*w)**(0.5)
    xi,yi,zi = u/norm,v/norm,w/norm
    x.append(xi)
    y.append(yi)
    z.append(zi)
fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
#ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
ax.scatter(x, y, z, s=100, c='r', zorder=10)
ax.set_title('Example of a uniformly sampled sphere', fontdict={'fontsize':20})
plt.show()


from geomstats.information_geometry.normal import NormalDistributions

normal = NormalDistributions()
fisher_metric = normal.metric

point_a = gs.array([1.0, 1.0])
point_b = gs.array([3.0, 1.0])

geodesic_ab_fisher = fisher_metric.geodesic(point_a, point_b)

n_points = 20
t = gs.linspace(0, 1, n_points)