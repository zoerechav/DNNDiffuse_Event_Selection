# pf_table.py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

from icecube import dataclasses

from pf_utils import (
    getDepth,
    erfc_fit,
)


def make_AddNewPassingFractions_Table(
    *,
    pf_folder,
    depth_space=(1.4, 2.1),
    shifts=(-3, -1, 0, 3, 10),
):
    """
    Factory for table-based atmospheric neutrino passing fractions.

    Parameters
    ----------
    pf_folder : str
        Directory containing PF tables.
    depth_space : tuple or array-like
        Allowed depth bins (km).
    shifts : tuple
        Shift values for PF systematics.
    """

    depth_space = np.asarray(depth_space)
    shifts = np.asarray(shifts)

    shift_labels = {-3: "_m3", -1: "_m1", 0: "_base", 3: "_p3", 10: "_p10"}
    fluxes = ["conv", "pr"]

    flav_dict = {
        12: "e",
        -12: "e",
        14: "mu",
        -14: "mu",
        16: "e",
        -16: "e",
    }

    def AddNewPassingFractions(frame):

        if "DNNDiffuse_v1.0.0_PF_features" not in frame:
            return
        
        if "DNNDiffuse_v1.0.0_pass" not in frame:
            return

        if not frame["DNNDiffuse_v1.0.0_pass"].value:
            return

        nu_type = int(frame['DNNDiffuse_v1.0.0_PF_features']['flavor'])
        if abs(nu_type) not in flav_dict:
            return

        pfs = dataclasses.I3MapStringDouble()

        cos_zenith = np.round(np.cos(frame['DNNDiffuse_v1.0.0_PF_features']['true_neutrino_zenith']), 2)
        energy = frame['DNNDiffuse_v1.0.0_PF_features']['true_neutrino_energy']

       
        if cos_zenith < 0:
            for flux in fluxes:
                for s in shifts:
                    pfs[f"PF_{flux}{shift_labels[s]}"] = 1.0

                pfs[f'erfc_{flux}_a'] = 1.0
                pfs[f'erfc_{flux}_b'] = 0.0
                pfs[f'erfc_{flux}_c'] = -1e10

            frame.Put("AtmNuPassingFraction", pfs)
            return

        
        depth_m = frame['DNNDiffuse_v1.0.0_PF_features']['depth_at_entry'] #m
        depth_km = np.round(depth_m / 1000.0, 2)

        depth_bin = np.round(
            depth_space[np.digitize(depth_km, depth_space) - 1],
            2,
        )

        for flux in fluxes:
            pfs_for_fit = np.empty(len(shifts))

            for i, s in enumerate(shifts):
                filename = (
                    f"{pf_folder}/"
                    f"PF{shift_labels[s]}_neut_type_{flux} "
                    f"{flav_dict[nu_type]}_at_depth_{depth_bin}km_at_None.npz" #filepath format
                )

                try:
                    f = np.load(filename)
                except FileNotFoundError:
                    pfs_for_fit[:] = np.nan
                    break

                interp_pf = RegularGridInterpolator(
                    (np.log10(f["energies"]), f["cos_zeniths"]),
                    f["PFs"],
                )

                PF = interp_pf((np.log10(energy), cos_zenith))

                pfs[f"PF_{flux}{shift_labels[s]}"] = float(PF)
                pfs_for_fit[i] = float(PF)

            if not np.all(np.isfinite(pfs_for_fit)):
                continue

            poly, _ = curve_fit(
                erfc_fit,
                shifts,
                pfs_for_fit,
                p0=[1, 5, 1],
                bounds=(0, np.inf),
                maxfev=20000,
            )

            pfs[f"erfc_{flux}_a"] = poly[0]
            pfs[f"erfc_{flux}_b"] = poly[1]
            pfs[f"erfc_{flux}_c"] = poly[2]

        frame.Put("AtmNuPassingFraction", pfs)

    return AddNewPassingFractions
