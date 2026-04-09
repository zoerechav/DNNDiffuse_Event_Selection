import numpy as np
from icecube import icetray
from icecube import dataclasses

# import your existing helpers
from utils.pf_features import getDepth, getDetectorTime


def add_depth(frame):
    """
    Add MCPrimaryDepth to the frame using PolyplopiaPrimary.
    """
    if "I3MCTree" in frame:
        if "PolyplopiaPrimary" in frame and "MCPrimaryDepth" not in frame:
            p = frame["PolyplopiaPrimary"]
            depth = getDepth(p)
            frame["MCPrimaryDepth"] = dataclasses.I3Double(depth)

    return True


def add_time(frame):
    """
    Add MCPrimaryTime to the frame using PolyplopiaPrimary.
    """
    if "I3MCTree" in frame:
        if "PolyplopiaPrimary" in frame and "MCPrimaryTime" not in frame:
            p = frame["PolyplopiaPrimary"]
            time = getDetectorTime(p)
            frame["MCPrimaryTime"] = dataclasses.I3Double(time)

    return True

def add_cc_tag(frame):
    if "I3MCTree" in frame:
        if "PolyplopiaPrimary" in frame and "MC_CCTag" not in frame:
            cc_tag = None
            p = frame['PolyplopiaPrimary']
            flavor = p.type
            
            itype = frame['I3MCWeightDict']['InteractionType']
            
            if itype == 1 and np.abs(flavor) == 14:
                cc_tag = True
                
            else:
                cc_tag = False
            
            frame['MC_CCTag']=icetray.I3Bool(cc_tag)