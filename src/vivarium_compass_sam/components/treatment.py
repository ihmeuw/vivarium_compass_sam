"""Prevention and treatment models"""
import pandas as pd

from vivarium.framework.engine import Builder

from vivarium_compass_sam.constants import data_keys, data_values, models
from vivarium_compass_sam.utilities import get_random_variable


class SQLNSTreatment:
    """Manages SQ-LNS prevention"""

    @property
    def name(self) -> str:
        """The name of this component."""
        return 'prevention_algorithm'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        draw = builder.configuration.input_data.input_draw_number
        self.randomness = builder.randomness.get_stream('initial_sq_lns_propensity')

        # todo get correct rrs
        self.severe_wasting_risk_ratio = get_random_variable(draw, *data_values.SQ_LNS.RISK_RATIO_WASTING_SEVERE)
        self.moderate_wasting_risk_ratio = get_random_variable(draw, *data_values.SQ_LNS.RISK_RATIO_WASTING_MODERATE)

        propensity_col = 'sq_lns_propensity'
        required_columns = [
            'age',
            propensity_col,
        ]

        self.propensity = builder.value.register_value_producer(
            data_keys.SQ_LNS.PROPENSITY,
            source=lambda index: self.population_view.get(index)[propensity_col],
            requires_columns=[propensity_col]
        )

        self.coverage = builder.value.register_value_producer(
            data_keys.SQ_LNS.COVERAGE,
            source=self.get_current_coverage,
            requires_columns=['age'],
            requires_values=[data_keys.SQ_LNS.PROPENSITY],
        )

        builder.value.register_value_modifier(
            'risk_factor.child_wasting.exposure_parameters',
            modifier=self.apply_wasting_prevention,
            requires_values=[data_keys.SQ_LNS.COVERAGE]
        )

        self.population_view = builder.population.get_view(required_columns)
        builder.population.initializes_simulants(self.on_initialize_simulants, creates_columns=[propensity_col],
                                                 requires_streams=['initial_sq_lns_propensity'])

    def on_initialize_simulants(self, pop_data):
        self.population_view.update(pd.Series(self.randomness.get_draw(pop_data.index), name='sq_lns_propensity'))

    def get_current_coverage(self, index: pd.Index) -> pd.Series:
        age = self.population_view.get(index)['age']
        propensity = self.propensity(index)

        coverage = ((propensity < data_values.SQ_LNS.COVERAGE_BASELINE)
                    & (data_values.SQ_LNS.COVERAGE_START_AGE <= age))

        return coverage

    # todo change to affect wasting
    def apply_wasting_prevention(self, index: pd.Index, target: pd.DataFrame) -> pd.Series:
        cat1_decrease = target.loc[:, 'cat1'] * (1 - self.severe_wasting_risk_ratio)
        cat2_decrease = target.loc[:, 'cat2'] * (1 - self.moderate_wasting_risk_ratio)

        covered = self.coverage(index)
        target.loc[covered, 'cat1'] = target.loc[covered, 'cat1'] - cat1_decrease.loc[covered]
        target.loc[covered, 'cat2'] = target.loc[covered, 'cat2'] - cat2_decrease.loc[covered]
        target.loc[covered, 'cat3'] = (target.loc[covered, 'cat3']
                                       + cat1_decrease.loc[covered] + cat2_decrease.loc[covered])
        return target
