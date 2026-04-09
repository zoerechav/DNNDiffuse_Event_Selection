# modules/energy.py

from icecube import dataclasses
from utils.deposited_energy import calc_deposit_energy

def add_deposited_energy(frame):
    """
    Adds: Deposited_Energy : I3Double (GeV)
    """
    energy = 0.0

    if "I3MCTree" in frame:
        mctree = frame["I3MCTree"]
        energy = calc_deposit_energy(mctree)

    frame["Deposited_Energy"] = dataclasses.I3Double(energy)
    return True