"""Loads, standardizes and validates input data for the simulation.

Abstract the extract and transform pieces of the artifact ETL.
The intent here is to provide a uniform interface around this portion
of artifact creation. The value of this interface shows up when more
complicated data needs are part of the project. See the BEP project
for an example.

`BEP <https://github.com/ihmeuw/vivarium_gates_bep/blob/master/src/vivarium_gates_bep/data/loader.py>`_

.. admonition::

   No logging is done here. Logging is done in vivarium inputs itself and forwarded.
"""
import numpy as np
import pandas as pd

from gbd_mapping import sequelae
from vivarium.framework.artifact import EntityKey
from vivarium_gbd_access import constants as gbd_constants
from vivarium_inputs import globals as vi_globals, interface, utilities as vi_utils, utility_data

from vivarium_compass_sam.constants import data_keys, data_values, metadata
from vivarium_compass_sam.data import utilities


def get_data(lookup_key: str, location: str) -> pd.DataFrame:
    """Retrieves data from an appropriate source.

    Parameters
    ----------
    lookup_key
        The key that will eventually get put in the artifact with
        the requested data.
    location
        The location to get data for.

    Returns
    -------
        The requested data.

    """
    mapping = {
        data_keys.POPULATION.LOCATION: load_population_location,
        data_keys.POPULATION.STRUCTURE: load_population_structure,
        data_keys.POPULATION.AGE_BINS: load_age_bins,
        data_keys.POPULATION.DEMOGRAPHY: load_demographic_dimensions,
        data_keys.POPULATION.TMRLE: load_theoretical_minimum_risk_life_expectancy,
        data_keys.POPULATION.ACMR: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.POPULATION.CRUDE_BIRTH_RATE: load_standard_data,

        data_keys.LRI.PREVALENCE: load_lri_prevalence,
        data_keys.LRI.INCIDENCE_RATE: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.LRI.REMISSION_RATE: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.LRI.DISABILITY_WEIGHT: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.LRI.EMR: load_lri_excess_mortality_rate,
        data_keys.LRI.CSMR: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.LRI.RESTRICTIONS: load_metadata,

        data_keys.PEM.MAM_DISABILITY_WEIGHT: load_pem_disability_weight,
        data_keys.PEM.SAM_DISABILITY_WEIGHT: load_pem_disability_weight,
        data_keys.PEM.EMR: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.PEM.CSMR: load_standard_gbd_2019_data_as_gbd_2020_data,
        data_keys.PEM.RESTRICTIONS: load_metadata,

        data_keys.WASTING.DISTRIBUTION: load_metadata,
        data_keys.WASTING.ALT_DISTRIBUTION: load_metadata,
        data_keys.WASTING.CATEGORIES: load_metadata,
        data_keys.WASTING.EXPOSURE: load_gbd_2020_exposure,
        data_keys.WASTING.RELATIVE_RISK: load_gbd_2020_rr,
        data_keys.WASTING.PAF: load_paf,
    }
    return mapping[lookup_key](lookup_key, location)


def load_population_location(key: str, location: str) -> str:
    if key != data_keys.POPULATION.LOCATION:
        raise ValueError(f'Unrecognized key {key}')

    return location


# noinspection PyUnusedLocal
def load_population_structure(key: str, location: str) -> pd.DataFrame:
    return interface.get_population_structure(location)


# noinspection PyUnusedLocal
def load_age_bins(key: str, location: str) -> pd.DataFrame:
    all_age_bins = (
        utilities.get_gbd_age_bins(metadata.GBD_2020_AGE_GROUPS)
        .set_index(['age_start', 'age_end', 'age_group_name'])
        .sort_index()
    )
    return all_age_bins


# noinspection PyUnusedLocal
def load_demographic_dimensions(key: str, location: str) -> pd.DataFrame:
    if key == data_keys.POPULATION.DEMOGRAPHY:
        return utilities.get_gbd_2020_demographic_dimensions()
    else:
        raise ValueError(f'Unrecognized key {key}')


# noinspection PyUnusedLocal
def load_theoretical_minimum_risk_life_expectancy(key: str, location: str) -> pd.DataFrame:
    return interface.get_theoretical_minimum_risk_life_expectancy()


def load_standard_data(key: str, location: str) -> pd.DataFrame:
    key = EntityKey(key)
    entity = utilities.get_entity(key)
    data = interface.get_measure(entity, key.measure, location).droplevel('location')
    return data


# noinspection PyUnusedLocal
def load_metadata(key: str, location: str):
    key = EntityKey(key)
    entity = utilities.get_entity(key)
    entity_metadata = entity[key.measure]
    if hasattr(entity_metadata, 'to_dict'):
        entity_metadata = entity_metadata.to_dict()
    return entity_metadata


# Project-specific data functions here

def load_standard_gbd_2019_data_as_gbd_2020_data(key: str, location: str) -> pd.DataFrame:
    gbd_2019_data = load_standard_data(key, location)
    return utilities.reshape_gbd_2019_data_as_gbd_2020_data(gbd_2019_data)


def load_lri_prevalence(key: str, location: str) -> pd.DataFrame:
    if key == data_keys.LRI.PREVALENCE:
        incidence_rate = get_data(data_keys.LRI.INCIDENCE_RATE, location)
        early_neonatal_prevalence = (incidence_rate[incidence_rate.index.get_level_values('age_start') == 0.0]
                                     * data_values.EARLY_NEONATAL_CAUSE_DURATION / 365)
        all_other_prevalence = (incidence_rate[incidence_rate.index.get_level_values('age_start') != 0.0]
                                * data_values.LRI_DURATION / 365)
        prevalence = pd.concat([early_neonatal_prevalence, all_other_prevalence])
        return prevalence
    else:
        raise ValueError(f'Unrecognized key {key}')


def load_lri_excess_mortality_rate(key: str, location: str) -> pd.DataFrame:
    if key == data_keys.LRI.EMR:
        csmr = get_data(data_keys.LRI.CSMR, location)
        prevalence = get_data(data_keys.LRI.PREVALENCE, location)
        data = (csmr / prevalence).fillna(0)
        data = data.replace([np.inf, -np.inf], 0)
        return data
    else:
        raise ValueError(f'Unrecognized key {key}')


def load_gbd_2020_exposure(key: str, location: str) -> pd.DataFrame:
    key = EntityKey(key)
    entity = utilities.get_gbd_2020_entity(key)

    data = utilities.get_data(key, entity, location, gbd_constants.SOURCES.EXPOSURE, 'rei_id',
                              metadata.GBD_2020_AGE_GROUPS, metadata.GBD_2020_ROUND_ID)
    data = utilities.process_exposure(data, key, entity, location, metadata.GBD_2020_AGE_GROUPS,
                                      metadata.GBD_2020_ROUND_ID)
    return data


def load_gbd_2020_rr(key: str, location: str) -> pd.DataFrame:
    key = EntityKey(key)
    entity = utilities.get_gbd_2020_entity(key)

    data = utilities.get_data(key, entity, location, gbd_constants.SOURCES.RR, 'rei_id', metadata.GBD_2020_AGE_GROUPS,
                              metadata.GBD_2020_ROUND_ID)
    data['rei_id'] = entity.gbd_id

    # from vivarium_inputs.extract.extract_relative_risk
    data = vi_utils.filter_to_most_detailed_causes(data)

    # from vivarium_inputs.core.get_relative_risk
    data = vi_utils.convert_affected_entity(data, 'cause_id')
    morbidity = data.morbidity == 1
    mortality = data.mortality == 1
    data.loc[morbidity & mortality, 'affected_measure'] = 'incidence_rate'
    data.loc[morbidity & ~mortality, 'affected_measure'] = 'incidence_rate'
    data.loc[~morbidity & mortality, 'affected_measure'] = 'excess_mortality_rate'
    data = utilities.filter_relative_risk_to_cause_restrictions(data)

    data = data.filter(vi_globals.DEMOGRAPHIC_COLUMNS + ['affected_entity', 'affected_measure', 'parameter']
                       + vi_globals.DRAW_COLUMNS)
    data = (data.groupby(['affected_entity', 'parameter'])
            .apply(utilities.normalize_age_and_years, fill_value=1, gbd_round_id=metadata.GBD_2020_ROUND_ID)
            .reset_index(drop=True))

    tmrel_cat = utility_data.get_tmrel_category(entity)
    tmrel_mask = data.parameter == tmrel_cat
    data.loc[tmrel_mask, vi_globals.DRAW_COLUMNS] = (data.loc[tmrel_mask, vi_globals.DRAW_COLUMNS]
                                                     .mask(np.isclose(data.loc[tmrel_mask, vi_globals.DRAW_COLUMNS],
                                                                      1.0), 1.0))

    data = utilities.validate_and_reshape_gbd_data(data, entity, key, location, metadata.GBD_2020_AGE_GROUPS,
                                                   metadata.GBD_2020_ROUND_ID)

    return data


def load_paf(key: str, location: str) -> pd.DataFrame:
    if key in [data_keys.WASTING.PAF]:
        risk = {
            data_keys.WASTING.PAF: data_keys.WASTING,
        }[key]

        exp = get_data(risk.EXPOSURE, location)
        rr = get_data(risk.RELATIVE_RISK, location)

        # paf = (sum_categories(exp * rr) - 1) / sum_categories(exp * rr)
        sum_exp_x_rr = (
            (exp * rr)
            .groupby(list(set(rr.index.names) - {'parameter'})).sum()
            .reset_index()
            .set_index(rr.index.names[:-1])
        )
        paf = (sum_exp_x_rr - 1) / sum_exp_x_rr
        return paf
    else:
        raise ValueError(f'Unrecognized key {key}')


def load_pem_disability_weight(key: str, location: str) -> pd.DataFrame:
    if key in [data_keys.PEM.MAM_DISABILITY_WEIGHT, data_keys.PEM.SAM_DISABILITY_WEIGHT]:
        key_sequelae_map = {
            data_keys.PEM.MAM_DISABILITY_WEIGHT: [sequelae.moderate_wasting_with_edema,
                                                  sequelae.moderate_wasting_without_edema],
            data_keys.PEM.SAM_DISABILITY_WEIGHT: [sequelae.severe_wasting_with_edema,
                                                  sequelae.severe_wasting_without_edema],
        }

        prevalence_disability_weight = []
        state_prevalence = []
        for s in key_sequelae_map[key]:
            sequela_prevalence = interface.get_measure(s, 'prevalence', location)
            sequela_disability_weight = interface.get_measure(s, 'disability_weight', location)

            prevalence_disability_weight += [sequela_prevalence * sequela_disability_weight]
            state_prevalence += [sequela_prevalence]

        gbd_2019_disability_weight = (
            (sum(prevalence_disability_weight) / sum(state_prevalence))
            .fillna(0)
            .droplevel('location')
        )
        disability_weight = utilities.reshape_gbd_2019_data_as_gbd_2020_data(gbd_2019_disability_weight)
        return disability_weight
    else:
        raise ValueError(f'Unrecognized key {key}')
