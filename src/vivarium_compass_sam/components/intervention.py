import pandas as pd

from vivarium.framework.engine import Builder

from vivarium_compass_sam.constants import data_keys, data_values, scenarios


class WastingTreatmentIntervention:

    def __init__(self):
        self.name = 'wasting_treatment_intervention'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        """Perform this component's setup."""
        self.scenario = scenarios.SCENARIOS[builder.configuration.intervention.scenario]
        self.clock = builder.time.clock()

        # NOTE: this operation is NOT commutative. This pipeline must not be modified anywhere else.
        builder.value.register_value_modifier(
            f'risk_factor.{data_keys.WASTING_TREATMENT.name}.exposure_parameters',
            modifier=self.coverage_effect,
        )

    # define a function to do the modification
    def coverage_effect(self, idx: pd.Index, target: pd.Series) -> pd.Series:
        # if this is the alternative scenario and the scale up has already started update coverage
        if self.scenario.has_alternative_treatment and data_values.SCALE_UP_START_DT <= self.clock():
            target['cat1'] = 0.1
            target['cat2'] = 0.0
            target['cat3'] = 0.9

        return target
