import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tqdm
import matplotlib.path as mpltPath
import shapely.geometry as geom
import os


# flux and livetime for weighting
NORM = 1e-18
GAMMA = -2.7
LIVETIME = 3.5e8  # seconds in 11 years

# overall DOM efficiency correction in ice models
DOMEFF = 1.015


FEATURES = ['reco_x',
            'reco_y',
            'reco_z',
            'reco_azi',
            'reco_zen',
            'reco_energy',
            'length',
            'q_tot',
            'ndof',
            'rlogl',
            'sqrresi',
            'chisqr',
            'detector_edge',
            'dist_to_string']
OTHERS = ['true_azi',
          'true_zen',
          'event',
          'run',
          'mu_bdt',
          'cs_bdt',
          'oa_diff_reco',
          'true_energy',
          'eg_err',
          'oangle',
          'weights',
          'primary_energy',
          'csky_ow']


def convex_hull(points):

    points = [(p[0],p[1]) for p in points]
    
    points = sorted(set(points))
     
    if len(points) <= 1:
        return points
 
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
 
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
 
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]

    return np.array(hull)


def GetDistance(xt=0,yt=0,xd=0,yd=0):
    surface_det_x = xd
    surface_det_y = yd
    x=[(surface_det_x[i],surface_det_y[i])for i in range(len(surface_det_x))]
    bound_2D=mpltPath.Path(x)
    line = geom.LineString(x)
    point = geom.Point(xt, yt)
    distval=point.distance(line)
    if not bound_2D.contains_points([(point.x, point.y)]):
        distval=-distval
    return(distval)


def dom_location():
    x = []
    y = []
    main_path = os.path.dirname(os.path.dirname(os.path.abspath( __file__ )))
    with open('/data/user/lseen/2025_LHAASO_Correlation_Analysis/sigmabdt/spreadsheets/Sunflower.txt', 'r') as file:
        for line in file:
            value = line.split()
            x.append(value[0])
            y.append(value[1])
    dom_loc = pd.DataFrame({'x':x, 'y':y}, dtype=float)
    return dom_loc


def find_shortest_distance(doms, x, y):
    
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    y_arr = np.atleast_1d(np.asarray(y, dtype=float))
    scalar_input = (x_arr.size == 1) and (y_arr.size == 1)

    domx = np.asarray(doms["x"], dtype=float)
    domy = np.asarray(doms["y"], dtype=float)

    # (Nevents, Ndoms)
    dx = x_arr[:, None] - domx[None, :]
    dy = y_arr[:, None] - domy[None, :]
    d2 = dx * dx + dy * dy

    dmin = np.sqrt(np.min(d2, axis=1))

    return float(dmin[0]) if scalar_input else dmin


def opening_angle(zen1, azi1, zen2, azi2):
    return  np.arccos(np.clip(
        np.cos(zen1)*np.cos(zen2) + np.sin(zen1)*np.sin(zen2)*np.cos(azi1 - azi2),
        -1, 1))


def get_weights(i3mcweightdict):
    oneweight = i3mcweightdict['OneWeight']
    eprim = i3mcweightdict['PrimaryNeutrinoEnergy']
    nevts = i3mcweightdict['NEvents']
    nfiles = np.size(np.unique(i3mcweightdict['Run']))
    return 10**(np.log10(oneweight) + GAMMA * (np.log10(eprim) - 5)
                + np.log10(NORM) - np.log10(nevts * nfiles)
                + np.log10(LIVETIME))
