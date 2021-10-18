from typing import NamedTuple

from vivarium_public_health.utilities import TargetString


#############
# Data Keys #
#############

METADATA_LOCATIONS = 'metadata.locations'


class __Population(NamedTuple):
    LOCATION: str = 'population.location'
    STRUCTURE: str = 'population.structure'
    AGE_BINS: str = 'population.age_bins'
    DEMOGRAPHY: str = 'population.demographic_dimensions'
    TMRLE: str = 'population.theoretical_minimum_risk_life_expectancy'
    ACMR: str = 'cause.all_causes.cause_specific_mortality_rate'
    CRUDE_BIRTH_RATE: str = 'covariate.live_births_by_sex.estimate'

    @property
    def name(self):
        return 'population'

    @property
    def log_name(self):
        return 'population'


POPULATION = __Population()


##########
# Causes #
##########


class __LowerRespiratoryInfections(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    PREVALENCE: TargetString = TargetString('cause.lower_respiratory_infections.prevalence')
    INCIDENCE_RATE: TargetString = TargetString('cause.lower_respiratory_infections.incidence_rate')
    REMISSION_RATE: TargetString = TargetString('cause.lower_respiratory_infections.remission_rate')
    DISABILITY_WEIGHT: TargetString = TargetString('cause.lower_respiratory_infections.disability_weight')
    EMR: TargetString = TargetString('cause.lower_respiratory_infections.excess_mortality_rate')
    CSMR: TargetString = TargetString('cause.lower_respiratory_infections.cause_specific_mortality_rate')
    RESTRICTIONS: TargetString = TargetString('cause.lower_respiratory_infections.restrictions')

    # Useful keys not for the artifact - distinguished by not using the colon type declaration

    @property
    def name(self):
        return 'lower_respiratory_infections'

    @property
    def log_name(self):
        return 'lower respiratory infections'


LRI = __LowerRespiratoryInfections()


class __ProteinEnergyMalnutrition(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    DISABILITY_WEIGHT: TargetString = TargetString('cause.protein_energy_malnutrition.disability_weight')
    EMR: TargetString = TargetString('cause.protein_energy_malnutrition.excess_mortality_rate')
    CSMR: TargetString = TargetString('cause.protein_energy_malnutrition.cause_specific_mortality_rate')
    RESTRICTIONS: TargetString = TargetString('cause.protein_energy_malnutrition.restrictions')

    # Useful keys not for the artifact - distinguished by not using the colon type declaration

    @property
    def name(self):
        return 'protein_energy_malnutrition'

    @property
    def log_name(self):
        return 'protein energy malnutrition'


PEM = __ProteinEnergyMalnutrition()


################
# Risk Factors #
################


class __Wasting(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    DISTRIBUTION: TargetString = 'risk_factor.child_wasting.distribution'
    ALT_DISTRIBUTION: TargetString = 'alternative_risk_factor.child_wasting.distribution'
    CATEGORIES: TargetString = 'risk_factor.child_wasting.categories'
    EXPOSURE: TargetString = 'risk_factor.child_wasting.exposure'
    RELATIVE_RISK: TargetString = 'risk_factor.child_wasting.relative_risk'
    PAF: TargetString = 'risk_factor.child_wasting.population_attributable_fraction'

    # Useful keys not for the artifact - distinguished by not using the colon type declaration
    CAT4 = 'cat4'
    CAT3 = 'cat3'
    CAT2 = 'cat2'
    CAT1 = 'cat1'

    @property
    def name(self):
        return 'child_wasting'

    @property
    def log_name(self):
        return 'child wasting'


WASTING = __Wasting()


class __SQLNS(NamedTuple):
    COVERAGE = 'sq_lns.coverage'
    PROPENSITY = 'sq_lns.propensity'

    @property
    def name(self):
        return 'sq_lns'

    @property
    def log_name(self):
        return 'sq-lns'


SQ_LNS = __SQLNS()


MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
    LRI,
    PEM,
    WASTING,
    SQ_LNS,
]
