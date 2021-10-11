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


class __DiarrhealDiseases(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    PREVALENCE: TargetString = TargetString('cause.diarrheal_diseases.prevalence')
    INCIDENCE_RATE: TargetString = TargetString('cause.diarrheal_diseases.incidence_rate')
    REMISSION_RATE: TargetString = TargetString('cause.diarrheal_diseases.remission_rate')
    DISABILITY_WEIGHT: TargetString = TargetString('cause.diarrheal_diseases.disability_weight')
    EMR: TargetString = TargetString('cause.diarrheal_diseases.excess_mortality_rate')
    CSMR: TargetString = TargetString('cause.diarrheal_diseases.cause_specific_mortality_rate')
    RESTRICTIONS: TargetString = TargetString('cause.diarrheal_diseases.restrictions')

    # Useful keys not for the artifact - distinguished by not using the colon type declaration

    @property
    def name(self):
        return 'diarrheal_diseases'

    @property
    def log_name(self):
        return 'diarrheal diseases'


DIARRHEA = __DiarrhealDiseases()


class __ProteinEnergyMalnutrition(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    MAM_DISABILITY_WEIGHT: TargetString = TargetString('sequela.moderate_acute_malnutrition.disability_weight')
    SAM_DISABILITY_WEIGHT: TargetString = TargetString('sequela.severe_acute_malnutrition.disability_weight')
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


class __WastingTreatment(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    EXPOSURE: TargetString = 'risk_factor.wasting_treatment.exposure'
    DISTRIBUTION: TargetString = 'risk_factor.wasting_treatment.distribution'
    CATEGORIES: TargetString = 'risk_factor.wasting_treatment.categories'
    RELATIVE_RISK: TargetString = 'risk_factor.wasting_treatment.relative_risk'
    PAF: TargetString = 'risk_factor.wasting_treatment.population_attributable_fraction'

    # Useful keys not for the artifact - distinguished by not using the colon type declaration
    TMREL_CATEGORY = 'cat2'
    COVERED_CATEGORIES = ['cat2', 'cat3']
    UNCOVERED_CATEGORIES = ['cat1']

    @property
    def name(self):
        return 'wasting_treatment'

    @property
    def log_name(self):
        return 'wasting treatment'


WASTING_TREATMENT = __WastingTreatment()


MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
    DIARRHEA,
    PEM,
    WASTING,
    WASTING_TREATMENT,
]
