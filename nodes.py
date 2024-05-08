import csv
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal, gaussian_kde
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as mcol
#import seaborn
from scipy.stats import gaussian_kde



sys.path.append("C:\\Users\\lenovo\\Documents\\physics\\voronoi\\decfirst")
import celldiv

argv = sys.argv

parser = argparse.ArgumentParser()

parser.add_argument("trajPath", type=str,
                    help="Trajectory path. Absolute or relative.")

parser.add_argument("-k", "--skip", type=int, required=False,
                    help="Trajectory frame skip rate. E.g. SKIP=10 will only \
                    render every 10th frame.",
                    default=1)

parser.add_argument("--min-cells", type=int, required=False,
                    help='Start rendering when system has at least this many cells',
                    default=1)

parser.add_argument("--inds", type=int, required=False, nargs='+',
                    help="Only render cells with these indices",
                    default=[])

parser.add_argument("-nf", "--num-frames", type=int, required=False,
                    help="Only render this many frames.",
                    default=sys.maxsize)

args = parser.parse_args(argv)

nSkip = args.skip


fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(0,40)
ax.set_ylim(0,40)
plt.title("2D scatter plot of a cell")


Files= ["CAP_1type.xyz" ]#, "CAP_2type.xyz"]
file_number=1
for filename in Files:
    with celldiv.TrajHandle(filename) as th:
        frameCount = 1
        try:
            for i in range(int(th.maxFrames/nSkip)):
                cell_com=[]
                frameCount += 1
                if frameCount > args.num_frames:
                    break
                f=th.ReadFrame(inc=nSkip )
                if len(args.inds) > 0:
                    f = [f[a] for a in args.inds]
                f = np.vstack(f)
                colors = np.arange(int(len(f)/192))
                for mi in range(int(len(f)/192)):
                    cell_com.append(np.mean(f[mi*192:(mi+1)*192],axis=0))
                    if  i==5:
                        x=[]
                        y=[]
                        #print(f[mi*192:mi*192 + 180])
                        x.append(f[mi*192:mi*192 + 180][:,0])
                        y.append(f[mi*192:mi*192 + 180][:,1])
                        # I want it to have a different color for each cell\
                        ax.scatter(x,y , c=colors[mi]*np.ones(180), cmap = 'jet')

        except celldiv.IncompleteTrajectoryError:
            print ("Stopping...") 





#3D plotting



plt.show()