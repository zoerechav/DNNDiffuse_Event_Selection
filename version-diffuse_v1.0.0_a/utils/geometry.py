# utils/geometry.py
import collections
import numpy as np
from matplotlib import path as mpltPath
from icecube import dataclasses, icetray, dataio, MuonGun


def select(geometry):
    """
    Select IceCube DOMs and sort per string by depth.
    """
    strings = collections.defaultdict(list)

    for omkey, omgeo in geometry.items():
        if np.iterable(omgeo):
            omgeo = omgeo[0]

        if omgeo.omtype == dataclasses.I3OMGeo.IceCube:
            strings[omkey.string].append((omkey, omgeo))

    for doms in strings.values():
        doms.sort(key=lambda x: x[1].position.z, reverse=True)

    return strings


def boundaries(geometry):
    """
    Return x,y boundary of detector using hard-coded IC86 strings.
    """
    strings = select(geometry)
    
    ##second outermost layer of strings
    manual_sides = [
        9, 10, 11, 12, 20, 29, 39, 49, 58, 66,
        65, 64, 71, 70, 69, 61, 52, 42, 32, 23,
        15, 8
    ]

    boundary_x = []
    boundary_y = []

    for side_string in manual_sides:
        pos = strings[side_string][0][1].position
        boundary_x.append(pos.x)
        boundary_y.append(pos.y)

    # close polygon
    boundary_x.append(boundary_x[0])
    boundary_y.append(boundary_y[0])

    return boundary_x, boundary_y


def get_surface_det_og(gcdFile):
    """
    contained boundary
    """
    surface_det = MuonGun.ExtrudedPolygon.from_file(gcdFile, padding=0)

    f = dataio.I3File(gcdFile)
    omgeo = f.pop_frame(icetray.I3Frame.Geometry)["I3Geometry"].omgeo

    surface_det_x, surface_det_y = boundaries(omgeo)
    
    x = [(surface_det_x[i], surface_det_y[i]) for i in range(len(surface_det_x))]

    bound_2D = mpltPath.Path(x)

    return bound_2D, surface_det


def get_surface_det(gcdFile, radius=700.0, npoints=1000):
    """
    partially contained outer boundary, a circle of radius 700m
    """
    surface_det = MuonGun.ExtrudedPolygon.from_file(
        gcdFile, padding=0
    )

    theta = np.linspace(0.0, 2.0 * np.pi, npoints, endpoint=False)

    surface_det_x = radius * np.cos(theta)
    surface_det_y = radius * np.sin(theta)

   
    x = [(surface_det_x[i], surface_det_y[i]) for i in range(len(surface_det_x))]

    # close polygon
    x.append(x[0])

    bound_2D = mpltPath.Path(x)

    return bound_2D, surface_det
