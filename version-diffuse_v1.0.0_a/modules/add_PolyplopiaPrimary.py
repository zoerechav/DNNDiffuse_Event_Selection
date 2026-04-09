from icecube import dataclasses


def add_PolyplopiaPrimary(frame):
    """
    Add PolyplopiaPrimary if missing and needed.

    Adds:
        PolyplopiaPrimary : I3Particle
    """

    if "I3MCTree" not in frame:
        return True

    if "PolyplopiaPrimary" not in frame:
        if 'MCPrimary' in frame:
            primary = frame["MCPrimary"]
            frame["PolyplopiaPrimary"] = dataclasses.I3Particle(primary)

    return True
