from icecube import dataclasses
from icecube.dataclasses import I3MapStringDouble


def map_snowstorm_parameters(frame):
    """
    Collect Snowstorm parameters into a flat I3MapStringDouble.

    Adds:
        SnowstormParameterDict : I3MapStringDouble
    """
    if 'SnowstormParameterDict' in frame:
        return True
    
    if (
        "SnowstormParametrizations" not in frame
        or "SnowstormParameters" not in frame
        or "SnowstormParameterRanges" not in frame
    ):
        return True

    parameter_map = {
        "HoleIceForward_Unified": ["HoleIceForward_Unified_p0", "HoleIceForward_Unified_p1"]
    }

    SnowstormParameterDict = I3MapStringDouble()

    for i, key in enumerate(frame["SnowstormParametrizations"]):
        parameters = frame["SnowstormParameters"]
        param_range = frame["SnowstormParameterRanges"][i]

        values = parameters[param_range.first:param_range.second]

        for j, v in enumerate(values):
            if key in parameter_map:
                SnowstormParameterDict[parameter_map[key][j]] = v
            else:
                SnowstormParameterDict[key] = v

    frame["SnowstormParameterDict"] = SnowstormParameterDict
    return True
