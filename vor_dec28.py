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
import seaborn

seaborn.set()

sys.path.append("C:\\Users\\lenovo\\Documents\\physics\\voronoi\\decfirst")
import celldiv


argv = sys.argv

if "--" not in argv:
    print("ERROR: No arguments provided to script")
    sys.exit(80)
else:
    a = argv.index("--")
    argv = argv[a + 1:]

helpString = """
Run as:
blender --background --python %s --
[options]
""" % __file__

parser = argparse.ArgumentParser(description=helpString)

parser.add_argument("trajPath", type=str,
                    help="Trajectory path. Absolute or relative.")

parser.add_argument("-s", "--smooth", action='store_true',
                    help="Do smoothing (really expensive and doesn't look as good)")

parser.add_argument("-k", "--skip", type=int, required=False,
                    help="Trajectory frame skip rate. E.g. SKIP=10 will only \
                    render every 10th frame.",
                    default=1)

parser.add_argument("-nc", "--noclear", type=bool, required=False,
                    help="specifying this will not clear the destination directory\
                    and restart rendering.",
                    default=False)

parser.add_argument("--min-cells", type=int, required=False,
                    help='Start rendering when system has at least this many cells',
                    default=1)

parser.add_argument("--inds", type=int, required=False, nargs='+',
                    help="Only render cells with these indices",
                    default=[])

parser.add_argument("-nf", "--num-frames", type=int, required=False,
                    help="Only render this many frames.",
                    default=sys.maxsize)

parser.add_argument("-r", "--res", type=int, default=1, required=False,
                    help='Renders images with resolution RES*1080p. RES>=1. \
                    Use 2 for 4k. A high number will devour your RAM.')

parser.add_argument("-cc", "--cell-color", type=int, nargs=3, required=False,
                    default=[72, 38, 153],
                    help="RGB values of cell color. From 0 to 255")

parser.add_argument("-bc", "--background-color", type=int, nargs=3,
                    required=False, default=[255,255,255],
                    help="RGB values of cell color. From 0 to 255")

parser.add_argument("-si", "--specular-intensity", type=float, required=False,
                    default = 0.0,
                    help="Set specular-intensity (shininess). From 0.0 to 1.0")

args = parser.parse_args(argv)

imageindex = 0
firstfaces = []


doSmooth = args.smooth
if doSmooth:
    print("Doing smoothing. Consider avoiding this feature...")


if (args.res < 1):
    print("ERROR: invalid resolution factor")
    sys.exit()

with open('C180_pentahexa.csv', newline='') as g:
    readerfaces = csv.reader(g, delimiter=',')
    for row in readerfaces:
        firstfaces.append([int(v) for v in row])


filename = os.path.realpath(args.trajPath)
basename = os.path.splitext(filename)[0] + "/images/CellDiv_"

nSkip = args.skip

if nSkip > 1:
    print("Rendering every %dth" % nSkip, "frame...")


noClear = args.noclear

sPath = os.path.splitext(filename)[0] + "/images/"

if not noClear and os.path.exists(sPath):
    for f in os.listdir(sPath):
        os.remove(sPath+f)

cellInds = []
minInd = args.min_cells - 1
if len(args.inds) > 0:
    minInd = max(minInd, min(args.inds))

stopAt = args.num_frames

# Set material color

#filename="inp.xyz"
#Files=["inp2.xyz" , "inp5.xyz" , "inp10.xyz"]
Files=["CM.xyz"]
cell_com=[]






def animation_func(xs,ys):
	plt.hist2d(xs,ys,bins=20)
	
def Plot_NN_disrtibution(vor):
    max=14
    N=np.zeros(max)
    a=np.arange(1,max+1)
    #vor.regions : list of indices of vertices for each voronoi cell
    # we can find how many neighbors each cell has by looking at the length of each list
    for j in vor.regions:
        N[len(j)-1]+=1
    #Dros=[0,0,0,0.03,0.28,0.46,0.2,0.03,0,0,0,0]
    #xenopus=[0,0,0.01,0.04,0.285,0.43,0.18,0.05,0.02,0,0,0]
    #plt.bar(a- 0.2,Dros,0.2, edgecolor='black',label="Drosophila(2171)")
    #plt.bar(a ,xenopus,0.2, edgecolor='black',label="Xenopus(1051)")
    #plt.bar(a + 0.2,N/sum(N),0.2, edgecolor='black',label="simulation(1783)")
    plt.bar(a,N/sum(N),0.2, edgecolor='black',label="simulation")

    
    plt.title("Number of Nearest Neighbors")
    plt.xlabel("$N_n$")
    plt.ylabel("fraction")
    #plt.legend()
    #cmap = plt.get_cmap('hot')
    #plt.set_cmap(cmap)
    plt.savefig("NN_plot{}.png".format(i))
    #plt.show()

def Plot_cell_distribution(cell_com):
    xs=[ i[0] for i in cell_com ]
    ys=[ i[1] for i in cell_com ]
    coords= np.array(cell_com)[: , 0:2]
    print(coords)
    vor = Voronoi(coords)
    fig = voronoi_plot_2d(vor)
    plt.show()

def Calculte_area_distribution(vor):
    s=[]
    for p in range(len(vor.points)):
        index_of_vor_region_for_point = vor.point_region[p]
        if -1 not in vor.regions[index_of_vor_region_for_point]:
            index_of_vertice = vor.regions[index_of_vor_region_for_point]
            S_seg=0
            for v in range(len(index_of_vertice)): 
                p1=vor.vertices[index_of_vertice[v]]
                p2=vor.vertices[index_of_vertice[v-1]]
                p3=vor.points[p]
                S_seg += 1/2*np.abs(( p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]) ))
                # plt.show()
                # plt.scatter(vor.vertices[ver][0],vor.vertices[ver][1])
                # print(vor.vertices[ver])
            if S_seg<=(np.pi):
                s.append(S_seg)
            S_seg=0
    return s

def Plot_area_distribution(vor,param):
    s=Calculte_area_distribution(vor)
    plt.hist(s,bins=200,label="Simulation")
    #plt.legend()
    plt.xlabel("Area")
    plt.ylabel("Count")
    plt.title("Area distribution for stiffnes {}".format(param))
    plt.savefig("Area_distribution{}.png".format(i))
    #plt.title("Area distribution for a simulated epithelium")
    #svg
    #print(s)        
        #plt.scatter(vor.points[p][0], vor.points[p][1] , marker=".")
        
def Color_area_distribution(vor):
    fig, ax = plt.subplots()
    s=Calculte_area_distribution(vor)
    Maximum=max(s) #find largest area - needed for sectioning 
    #18 Jan
    # Make a user-defined colormap.
    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
    # Make a normalizer that will map the time values from
    # [start_time,end_time+1] -> [0,1].
    cnorm = mcol.Normalize(vmin=0,vmax=np.pi)
    # Turn these into an object that can be used to map time values to colors and
    # can be passed to plt.colorbar().
    cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
    cpick.set_array([])

    for p in range(len(vor.points)):
        index_of_vor_region_for_point = vor.point_region[p]
        if -1 not in vor.regions[index_of_vor_region_for_point]:
            index_of_vertice = vor.regions[index_of_vor_region_for_point]
            S_seg=0
            A_seg=0
            for v in range(len(index_of_vertice)): 
                p1=vor.vertices[index_of_vertice[v]]
                p2=vor.vertices[index_of_vertice[v-1]]
                p3=vor.points[p]
                S_seg += 1/2*np.abs(( p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]) ))
                A_seg += np.abs((p1[0]-p2[0]) + (p1[1]-p2[1])*1j)
                # plt.show()
                #plt.scatter(vor.vertices[ver][0],vor.vertices[ver][1])
                #print(vor.vertices[ver])
            
            if S_seg<=(np.pi) and A_seg<2*np.pi:
                p = []
                for ind in index_of_vertice:
                    p.append(vor.vertices[ind])
                patches = []
                patches.append(Polygon(p, closed=True))
                p = PatchCollection(patches,edgecolor="r",alpha=0.8)
                p.set_color(cpick.to_rgba(S_seg))
                ax.add_collection(p)
    #ax.autoscale()
    ax.set_xlim([-5,45])
    ax.set_ylim([-5,45])
    
    # F = plt.figure()
    # A = F.add_subplot(111)
    # for y, t in zip(ydat,tim):
    #     A.plot(xdat,y,color=cpick.to_rgba(t))

    plt.colorbar(cpick,label="Area",ax=plt.gca())
    plt.savefig("Area_plot{}.png".format(i))
    plt.close()

    # fig.colorbar(ax=ax)
    # plt.show()

    # bottom = cm.get_cmap('Blues', 128)
    # newcolors = np.vstack((bottom(np.linspace(0, 1, 128))))
    # newcmp = ListedColormap(newcolors, name='OrangeBlue')

    # sm = plt.cm.ScalarMappable(cmap=newcmp, norm=plt.Normalize(vmin=0, vmax=Maximum))
    # sm.set_array([])
    # plt.colorbar(sm,ax=ax)
    #plt.show()
    
def Find_radial_distribution(vor,initial_center):
    x_c=initial_cell[0]
    y_c=initial_cell[1]
    radial_dist=10
    size_Box=100 #Insert this by hand
    Area_sections=[]
    for _ in range(int(size_Box/(2*radial_dist))+3):
        Area_sections.append([])
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True)
    for p in range(len(vor.points)):
        coord_each_point=vor.points[p]
        dist=np.abs((x_c-coord_each_point[0]) + (y_c-coord_each_point[1])*1j)
        

        index_of_vor_region_for_point = vor.point_region[p]
        if -1 not in vor.regions[index_of_vor_region_for_point]:
            index_of_vertice = vor.regions[index_of_vor_region_for_point]
            S_seg=0
            for v in range(len(index_of_vertice)): 
                p1=vor.vertices[index_of_vertice[v]]
                p2=vor.vertices[index_of_vertice[v-1]]
                p3=vor.points[p]
                S_seg += 1/2*np.abs(( p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]) ))
        if S_seg<=(np.pi):
            Area_sections[int(dist/radial_dist)].append(S_seg)
    #ax1.title("Area distribution for different radii")  
    fig.suptitle('Area distribution at different radii', fontsize=16)    
    ax1.hist(Area_sections[0],bins=70)
    ax1.set_title('Area distribution of cells d=0-10 from the initial cell')
    ax2.hist(Area_sections[1],bins=70)
    ax2.set_title('Area distribution of cells d=10-20 from the initial cell')
    ax3.hist(Area_sections[2],bins=70)
    ax3.set_title('Area distribution of cells d=20-30 from the initial cell')
    ax4.hist(Area_sections[3],bins=70)
    ax4.set_title('Area distribution of cells d=30-40 from the initial cell')
    

    #ax5.hist(Area_sections[4],bins=50)
    ax4.set_xlabel("Area")
    
    #for a in range(len(Area_sections)):
    #    if Area_sections[a]:
            # print(type(a[0]))
            #plt.hist(Area_sections[a],bins=50,density=True,alpha=0.7,label="Radius {}".format((a+1)*radial_dist))
            #plt.hist(Area_sections[a],bins=50,alpha=0.7)
            #plt.legend()



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
                if len(f) < minInd+1:
                    print("Only ", len(f), "cells in frame ", th.currFrameNum,
                        " skipping...")
                    continue
                if len(args.inds) > 0:
                    f = [f[a] for a in args.inds]
                f = np.vstack(f)
                for mi in range(int(len(f)/192)):
                    cell_com.append(np.mean(f[mi*192:(mi+1)*192],axis=0))
                
                coords= np.array(cell_com)[: , 0:2] #2D coordinates
                
                #print(coords)
                if i == 0:
                    initial_cell = coords[0]
                
                if i > 30:
                    vor = Voronoi(coords)
                    #print(len(vor.points))
                    #Color_area_distribution(vor)
                    #Find_radial_distribution(vor,initial_cell)
                
                    #Plot_NN_disrtibution(vor)
                   
                    Plot_area_distribution(vor,0.6)
                #NN dist
                
                #Plot_NN_disrtibution(vor)
                #Color_area_distribution(vor)
            #print(vor.vertices)
            #print(vor.regions)
            #Plot_cell_distribution(coords) 

        except celldiv.IncompleteTrajectoryError:
            print ("Stopping...")   
plt.show()                
