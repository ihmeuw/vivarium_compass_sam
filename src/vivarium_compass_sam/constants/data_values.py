from datetime import datetime
from typing import Dict, NamedTuple, Tuple

from vivarium_compass_sam.utilities import get_norm_from_quantiles, get_lognorm_from_quantiles

#######################
# Universal Constants #
#######################
YEAR_DURATION: float = 365.25


##########################
# Cause Model Parameters #
##########################

# LRI duration in days
LRI_DURATION: int = 10

# duration > bin_duration, so there is effectively no remission,
# and duration within the bin is bin_duration / 2
EARLY_NEONATAL_CAUSE_DURATION: float = 3.5


########################
# Prevention Constants #
########################
class __SQLNS(NamedTuple):
    COVERAGE_START_AGE: float = 0.5
    COVERAGE_BASELINE: float = 0.0
    COVERAGE_RAMP_UP: float = 0.9
    RISK_RATIO_WASTING_SEVERE: Tuple = ('sq_lns_severe_wasting_effect',
                                        get_lognorm_from_quantiles(median=0.85, lower=0.74, upper=0.98))
    RISK_RATIO_WASTING_MODERATE: Tuple = ('sq_lns_moderate_wasting_effect',
                                          get_lognorm_from_quantiles(median=0.82, lower=0.74, upper=0.91))


SQ_LNS = __SQLNS()


###################################
# Scale-up Intervention Constants #
###################################
SCALE_UP_START_DT = datetime(2023, 1, 1)
