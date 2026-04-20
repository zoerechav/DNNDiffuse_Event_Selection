"""
Final-level cut values for DNNDiffuse v1_0_0.
"""

# config/cuts.py

#z and energy values
Z_LOWER = -700.0
Z_UPPER =  500.0

Z_CONTAINED = 500.0
ENERGY_CONTAINED_Z = 1e4  # GeV

MIN_RECO_ENERGY = 1e3     # GeV


DUST_MIN = -150.0
DUST_MAX =  -50.0

#bdt values
CASCADE_BDT_MIN = 0.1
MUON_BDT_MAX    = 5e-3

#qtot values
CONTAINED_QTOT_MIN = 100
PARTIAL_QTOT_MIN = 200

#theo's dnn values
THEO_SCORE_MIN = 0.1

#cosmic ray cut values
ZENITH_MIN = 0.8
ENERGY_MIN = 1e4
Z_MIN = 100


#energy dependent geometry cuts
#'bottom_slice' --> uncontained events below 10 tev and belove -500 m
#'uncontained; --> all uncontained events below 10 tev
ENERGY_Z_MODE = "uncontained"

#use cut or not
USE_DUST_CUT = True
USE_BDT_CUTS = True
USE_ENERGY_Z_CUT = True
USE_QTOT_CUTS = True
USE_THEO_CUTS = True
USE_COSMIC_RAY_CUTS = True

