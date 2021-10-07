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

# diarrhea duration in days
DIARRHEA_DURATION: int = 10

# measles duration in days
MEASLES_DURATION: int = 10

# LRI duration in days
LRI_DURATION: int = 10

# duration > bin_duration, so there is effectively no remission,
# and duration within the bin is bin_duration / 2
EARLY_NEONATAL_CAUSE_DURATION: float = 3.5


############################
# Wasting Model Parameters #
############################
class __Wasting(NamedTuple):
    # Wasting age start (in years)
    START_AGE: float = 0.5

    # Wasting treatment distribution type and categories
    DISTRIBUTION: str = 'ordered_polytomous'
    CATEGORIES: Dict[str, str] = {
        'cat1': 'Untreated',
        'cat2': 'Baseline treatment',
        'cat3': 'Alternative scenario treatment',
    }

    # Wasting treatment coverage
    COVERAGE_START_AGE: float = 28 / YEAR_DURATION  # ~0.0767
    BASELINE_TX_COVERAGE: Tuple = ('sam_tx_coverage', get_norm_from_quantiles(mean=0.488, lower=0.374, upper=0.604))
    ALTERNATIVE_TX_COVERAGE: float = 0.9

    # Wasting treatment efficacy
    BASELINE_SAM_TX_EFFICACY: Tuple = ('sam_tx_efficacy', get_norm_from_quantiles(mean=0.700, lower=0.64, upper=0.76))
    BASELINE_MAM_TX_EFFICACY: Tuple = ('mam_tx_efficacy', get_norm_from_quantiles(mean=0.731, lower=0.585, upper=0.877))
    SAM_TX_ALTERNATIVE_EFFICACY: float = 0.75
    MAM_TX_ALTERNATIVE_EFFICACY: float = 0.75

    # Incidence correction factor (total exit rate)
    SAM_K: float = ('sam_incidence_correction', get_lognorm_from_quantiles(median=6.7, lower=5.3, upper=8.4))

    # Untreated time to recovery in days
    MAM_UX_RECOVERY_TIME: float = 63.0
    MILD_WASTING_UX_RECOVERY_TIME: float = 1000.0

    # Treated time to recovery in days
    SAM_TX_RECOVERY_TIME_OVER_6MO: float = 48.3
    SAM_TX_RECOVERY_TIME_UNDER_6MO: float = 13.3
    MAM_TX_RECOVERY_TIME_OVER_6MO: float = 41.3
    MAM_TX_RECOVERY_TIME_UNDER_6MO: float = 13.3


WASTING = __Wasting()


###########################
# Maternal BMI Parameters #
###########################
class __MaternalBMI(NamedTuple):
    DISTRIBUTION: str = 'dichotomous'
    CATEGORIES: Dict[str, str] = {
        'cat1': 'BMI < 18.5',
        'cat2': 'BMI >= 18.5',
    }

    EXPOSURE: Tuple = ('maternal_bmi_exposure', get_norm_from_quantiles(mean=0.224, lower=0.217, upper=0.231))


MATERNAL_BMI = __MaternalBMI()


######################################
# Treatment and Prevention Constants #
######################################
class __SQLNS(NamedTuple):
    COVERAGE_START_AGE: float = 0.5
    COVERAGE_BASELINE: float = 0.0
    COVERAGE_RAMP_UP: float = 0.9
    RISK_RATIO_WASTING: Tuple = ('sq_lns_wasting_effect',
                                 get_lognorm_from_quantiles(median=0.82, lower=0.74, upper=0.91))
    RISK_RATIO_STUNTING_SEVERE: Tuple = ('sq_lns_severe_stunting_effect',
                                         get_lognorm_from_quantiles(median=0.85, lower=0.74, upper=0.98))
    RISK_RATIO_STUNTING_MODERATE: Tuple = ('sq_lns_moderate_stunting_effect',
                                           get_lognorm_from_quantiles(median=0.93, lower=0.88, upper=0.98))


SQ_LNS = __SQLNS()


###################################
# Scale-up Intervention Constants #
###################################

SCALE_UP_START_DT = datetime(2023, 1, 1)
