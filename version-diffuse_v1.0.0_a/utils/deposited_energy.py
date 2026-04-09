from icecube.icetray import I3Tray
from icecube import icetray, dataio, dataclasses, simclasses, phys_services
from icecube.icetray import I3Units


##helper function authored by Tianlu Yuan
def calc_deposit_energy(mctree):
    """
    Compute deposited energy in ice from cascade particles.
    Returns energy in GeV.
    """
    losses = 0.0
    for p in mctree:
        if not p.is_cascade:
            continue
        
        if not p.location_type == dataclasses.I3Particle.InIce:
            continue
            
        if p.shape == p.Dark:
            continue
            
        if p.type in (p.Hadrons,p.PiPlus,p.PiMinus,p.NuclInt):
            if p.energy < 1 * I3Units.GeV:
                losses += 0.8 * p.energy
            else:
                energyScalingFactor = 1.0 + ((p.energy / I3Units.GeV / 0.399) ** -0.130) * (0.467 - 1.0)
                losses += energyScalingFactor * p.energy
        else:
            losses += p.energy

    return losses