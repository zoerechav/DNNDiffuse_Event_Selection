from icecube import icetray, dataclasses
from utils.geometry import (get_surface_det,get_surface_det_og)

def make_boundary_check(gcdFile):
    """
    Factory that returns a frame function with cached geometry.
    """

    ##cached after first call
    bound_2D_og, surface_det_og = get_surface_det_og(gcdFile) ##bounded by second outermost string layer
    bound_2D,    surface_det   = get_surface_det(gcdFile) ##bounded by circle of radius 700m

    def boundary_check(frame):

        inlimit = False
        big_inlimit = False

        if 'PreferredFit' in frame:
            particle = frame['PreferredFit']

            if (
                (particle.pos.z <= 500.0)
                and (particle.pos.z >= -500.0)
                and bound_2D_og.contains_points([(particle.pos.x, particle.pos.y)])):
                
                inlimit = True

            if inlimit == False:
                if (
                    (particle.pos.z <= 500.0)
                    and (particle.pos.z >= -700.0)
                    and bound_2D.contains_points([(particle.pos.x, particle.pos.y)])):
                    big_inlimit = True

        frame["contained"] = icetray.I3Bool(inlimit)
        frame["partial"]   = icetray.I3Bool(big_inlimit)

        return True

    return boundary_check
