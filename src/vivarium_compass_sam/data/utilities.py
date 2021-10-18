from itertools import product
from numbers import Real
from typing import List, Union
import warnings

import pandas as pd

from gbd_mapping import causes, covariates, risk_factors, Cause, ModelableEntity, RiskFactor
from vivarium.framework.artifact import EntityKey
from vivarium_gbd_access import constants as gbd_constants, gbd
from vivarium_gbd_access.utilities import get_draws, query
from vivarium_inputs import globals as vi_globals, utilities as vi_utils, utility_data
from vivarium_inputs.mapping_extension import alternative_risk_factors, AlternativeRiskFactor
from vivarium_inputs.validation.raw import check_metadata
from vivarium_inputs.validation.sim import validate_for_simulation

from vivarium_compass_sam.constants.metadata import ARTIFACT_INDEX_COLUMNS, GBD_2020_AGE_GROUPS, GBD_2020_ROUND_ID


def _load_em_from_meid(location, meid, measure):
    location_id = utility_data.get_location_id(location)
    data = gbd.get_modelable_entity_draws(meid, location_id)
    data = data[data.measure_id == vi_globals.MEASURES[measure]]
    data = vi_utils.normalize(data, fill_value=0)
    data = data.filter(vi_globals.DEMOGRAPHIC_COLUMNS + vi_globals.DRAW_COLUMNS)
    data = vi_utils.reshape(data)
    data = vi_utils.scrub_gbd_conventions(data, location)
    data = vi_utils.split_interval(data, interval_column='age', split_column_prefix='age')
    data = vi_utils.split_interval(data, interval_column='year', split_column_prefix='year')
    return vi_utils.sort_hierarchical_data(data)


def get_entity(key: EntityKey) -> ModelableEntity:
    # Map of entity types to their gbd mappings.
    type_map = {
        'cause': causes,
        'covariate': covariates,
        'risk_factor': risk_factors,
        'alternative_risk_factor': alternative_risk_factors
    }
    return type_map[key.type][key.name]


def reshape_gbd_2019_data_as_gbd_2020_data(gbd_2019_data: pd.DataFrame) -> pd.DataFrame:
    # Get target output index
    full_gbd_2020_idx = get_gbd_2020_demographic_dimensions().index

    # Get target index subset to GBD 2019 estimation years
    subset_gbd_2019_years_idx = (
        full_gbd_2020_idx[full_gbd_2020_idx.get_level_values('year_start') < 2020]
        .droplevel('age_end')
        .reorder_levels(['year_start', 'year_end', 'sex', 'age_start'])
    )

    # Reindex data with GBD 2020 age bins across GBD 2019 estimation years and fill forward NAs
    gbd_2019_years_gbd_2020_age_bins_data = (
        gbd_2019_data
        .droplevel('age_end')
        .reorder_levels(['year_start', 'year_end', 'sex', 'age_start'])
        .reindex(index=subset_gbd_2019_years_idx)
        .sort_index()
        .ffill()
    )

    # Get full target index excluding year end and age end columns
    full_gbd_2020_idx_without_end_columns = (
        full_gbd_2020_idx
        .droplevel(['year_end', 'age_end'])
        .reorder_levels(['sex', 'age_start', 'year_start'])
    )

    # Reindex data with GBD 2020 estimation years and fill forward NAs
    full_data_without_end_columns = (
        gbd_2019_years_gbd_2020_age_bins_data
        .droplevel('year_end')
        .reorder_levels(['sex', 'age_start', 'year_start'])
        .reindex(index=full_gbd_2020_idx_without_end_columns)
        .sort_index()
        .ffill()
        .reset_index()
    )

    # Repopulate year_end and age_end columns and set index
    full_data = apply_artifact_index(full_data_without_end_columns)
    return full_data


def get_gbd_2020_entity(key: str) -> ModelableEntity:
    # from load_standard_data
    entity = get_entity(key)

    if isinstance(entity, RiskFactor) or isinstance(entity, Cause):
        # Set risk factor age restrictions for GBD 2020
        if 'yll_age_group_id_start' in entity.restrictions:
            entity.restrictions.yll_age_group_id_start = min(GBD_2020_AGE_GROUPS)
        if 'yld_age_group_id_start' in entity.restrictions:
            entity.restrictions.yld_age_group_id_start = min(GBD_2020_AGE_GROUPS)
        if 'yll_age_group_id_end' in entity.restrictions:
            entity.restrictions.yll_age_group_id_end = max(GBD_2020_AGE_GROUPS)
        if 'yld_age_group_id_end' in entity.restrictions:
            entity.restrictions.yld_age_group_id_end = max(GBD_2020_AGE_GROUPS)

    return entity


def get_data(key: EntityKey, entity: ModelableEntity, location: str, source: str, gbd_id_type: str,
             age_group_ids: List[int], gbd_round_id: int, decomp_step: str = 'iterative') -> pd.DataFrame:
    # from interface.get_measure
    # from vivarium_inputs.core.get_data
    location_id = utility_data.get_location_id(location) if isinstance(location, str) else location

    # from vivarium_inputs.core.get_{measure}
    # from vivarium_inputs.extract.extract_data
    check_metadata(entity, key.measure)

    # from vivarium_inputs.extract.extract_{measure}
    # from vivarium_gbd_access.gbd.get_{measure}
    data = get_draws(gbd_id_type=gbd_id_type,
                     gbd_id=entity.gbd_id,
                     source=source,
                     location_id=location_id,
                     sex_id=gbd_constants.SEX.MALE + gbd_constants.SEX.FEMALE,
                     age_group_id=age_group_ids,
                     gbd_round_id=gbd_round_id,
                     decomp_step=decomp_step,
                     status='best')
    return data


def validate_and_reshape_gbd_data(data: pd.DataFrame, entity: ModelableEntity, key: EntityKey,
                                  location: str, age_group_ids: List[int], gbd_round_id: int) -> pd.DataFrame:

    # from vivarium_inputs.core.get_data
    data = vi_utils.reshape(data, value_cols=vi_globals.DRAW_COLUMNS)

    # from interface.get_measure
    data = _scrub_gbd_conventions(data, location, age_group_ids)

    estimation_years = get_gbd_estimation_years(gbd_round_id)
    validation_years = pd.DataFrame({'year_start': range(min(estimation_years), max(estimation_years) + 1)})
    validation_years['year_end'] = validation_years['year_start'] + 1

    validate_for_simulation(data, entity, key.measure, location, years=validation_years,
                            age_bins=get_gbd_age_bins(age_group_ids))
    data = vi_utils.split_interval(data, interval_column='age', split_column_prefix='age')
    data = vi_utils.split_interval(data, interval_column='year', split_column_prefix='year')
    data = vi_utils.sort_hierarchical_data(data).droplevel('location')
    return data


def normalize_age_and_years(data: pd.DataFrame, fill_value: Real = None,
                            cols_to_fill: List[str] = vi_globals.DRAW_COLUMNS,
                            gbd_round_id: int = GBD_2020_ROUND_ID) -> pd.DataFrame:
    data = vi_utils.normalize_sex(data, fill_value, cols_to_fill)

    # vi_inputs.normalize_year(data)
    binned_years = get_gbd_estimation_years(gbd_round_id)
    years = {'annual': list(range(min(binned_years), max(binned_years) + 1)), 'binned': binned_years}

    if 'year_id' not in data:
        # Data doesn't vary by year, so copy for each year.
        df = []
        for year in years['annual']:
            fill_data = data.copy()
            fill_data['year_id'] = year
            df.append(fill_data)
        data = pd.concat(df, ignore_index=True)
    elif set(data.year_id) == set(years['binned']):
        data = vi_utils.interpolate_year(data)
    else:  # set(data.year_id.unique()) == years['annual']
        pass

    # Dump extra data.
    data = data[data.year_id.isin(years['annual'])]

    data = vi_utils.normalize_age(data, fill_value, cols_to_fill)
    return data


def get_gbd_2020_demographic_dimensions() -> pd.DataFrame:
    estimation_years = get_gbd_estimation_years(GBD_2020_ROUND_ID)
    year_starts = range(estimation_years[0], estimation_years[-1] + 1)
    age_bins = get_gbd_age_bins(GBD_2020_AGE_GROUPS)

    unique_index_data = (pd.DataFrame(product(['Female', 'Male'], age_bins.age_start, year_starts))
                         .rename(columns={0: 'sex', 1: 'age_start', 2: 'year_start'}))

    index_data = apply_artifact_index(unique_index_data)
    return index_data


def apply_artifact_index(data: pd.DataFrame) -> pd.DataFrame:
    """Sets data frame index to match artifact format.
     Populates year_end and age_end columns if they are missing"""

    if 'year_end' not in data.columns:
        data['year_end'] = data['year_start'] + 1
    if 'age_end' not in data.columns:
        age_bins = get_gbd_age_bins(GBD_2020_AGE_GROUPS)
        data['age_end'] = data['age_start'].apply(lambda x: {start: end for start, end
                                                             in zip(age_bins.age_start, age_bins.age_end)}[x])
    data = data.set_index(ARTIFACT_INDEX_COLUMNS)
    return data


def get_gbd_estimation_years(gbd_round_id: int) -> List[int]:
    """Gets the estimation years for a particular gbd round."""
    from db_queries import get_demographics
    warnings.filterwarnings("default", module="db_queries")

    return get_demographics(gbd_constants.CONN_DEFS.EPI, gbd_round_id=gbd_round_id)['year_id']


def _scrub_gbd_conventions(data: pd.DataFrame, location: str, age_group_ids: List[int]) -> pd.DataFrame:
    data = vi_utils.scrub_location(data, location)
    data = vi_utils.scrub_sex(data)
    data = _scrub_age(data, age_group_ids)
    data = vi_utils.scrub_year(data)
    data = vi_utils.scrub_affected_entity(data)
    return data


def process_exposure(data: pd.DataFrame, key: str, entity: Union[RiskFactor, AlternativeRiskFactor],
                     location: str, age_group_ids: List[int], gbd_round_id: int) -> pd.DataFrame:
    data['rei_id'] = entity.gbd_id

    # from vivarium_inputs.extract.extract_exposure
    allowable_measures = [vi_globals.MEASURES['Proportion'], vi_globals.MEASURES['Continuous'],
                          vi_globals.MEASURES['Prevalence']]
    proper_measure_id = set(data.measure_id).intersection(allowable_measures)
    if len(proper_measure_id) != 1:
        raise vi_globals.DataAbnormalError(f'Exposure data have {len(proper_measure_id)} measure id(s). '
                                           f'Data should have exactly one id out of {allowable_measures} '
                                           f'but came back with {proper_measure_id}.')
    data = data[data.measure_id == proper_measure_id.pop()]

    # from vivarium_inputs.core.get_exposure
    data = data.drop('modelable_entity_id', axis='columns')

    if entity.name in vi_globals.EXTRA_RESIDUAL_CATEGORY:
        # noinspection PyUnusedLocal
        cat = vi_globals.EXTRA_RESIDUAL_CATEGORY[entity.name]
        data = data.drop(labels=data.query('parameter == @cat').index)
        data[vi_globals.DRAW_COLUMNS] = data[vi_globals.DRAW_COLUMNS].clip(lower=vi_globals.MINIMUM_EXPOSURE_VALUE)

    if entity.distribution in ['dichotomous', 'ordered_polytomous', 'unordered_polytomous']:
        tmrel_cat = utility_data.get_tmrel_category(entity)
        exposed = data[data.parameter != tmrel_cat]
        unexposed = data[data.parameter == tmrel_cat]
        #  FIXME: We fill 1 as exposure of tmrel category, which is not correct.
        data = pd.concat([normalize_age_and_years(exposed, fill_value=0, gbd_round_id=gbd_round_id),
                          normalize_age_and_years(unexposed, fill_value=1, gbd_round_id=gbd_round_id)],
                         ignore_index=True)

        # normalize so all categories sum to 1
        cols = list(set(data.columns).difference(vi_globals.DRAW_COLUMNS + ['parameter']))
        data = data.set_index(cols + ['parameter'])
        sums = (
            data.groupby(cols)[vi_globals.DRAW_COLUMNS].sum()
                .reindex(index=data.index)
        )
        data = data.divide(sums).reset_index()
    else:
        data = vi_utils.normalize(data, fill_value=0)

    data = data.filter(vi_globals.DEMOGRAPHIC_COLUMNS + vi_globals.DRAW_COLUMNS + ['parameter'])
    data = validate_and_reshape_gbd_data(data, entity, key, location, age_group_ids, gbd_round_id)
    return data


def _scrub_age(data: pd.DataFrame, age_group_ids: List[int]) -> pd.DataFrame:
    if 'age_group_id' in data.index.names:
        age_bins = get_gbd_age_bins(age_group_ids).set_index('age_group_id')
        id_levels = data.index.levels[data.index.names.index('age_group_id')]
        interval_levels = [pd.Interval(age_bins.age_start[age_id], age_bins.age_end[age_id], closed='left')
                           for age_id in id_levels]
        data.index = data.index.rename('age', 'age_group_id').set_levels(interval_levels, 'age')
    return data


def get_gbd_age_bins(age_group_ids: List[int]) -> pd.DataFrame:
    # from gbd.get_age_bins()
    q = f"""
                SELECT age_group_id,
                       age_group_years_start,
                       age_group_years_end,
                       age_group_name
                FROM age_group
                WHERE age_group_id IN ({','.join([str(a) for a in age_group_ids])})
                """
    raw_age_bins = query(q, 'shared')

    # from utility_data.get_age_bins()
    age_bins = (
        raw_age_bins[['age_group_id', 'age_group_name', 'age_group_years_start', 'age_group_years_end']]
        .rename(columns={'age_group_years_start': 'age_start', 'age_group_years_end': 'age_end'})
    )

    # set age start for birth prevalence age bin to -1 to avoid validation issues
    age_bins.loc[age_bins['age_end'] == 0.0, 'age_start'] = -1.0
    return age_bins


def filter_relative_risk_to_cause_restrictions(data: pd.DataFrame) -> pd.DataFrame:
    """ It applies age restrictions according to affected causes
    and affected measures. If affected measure is incidence_rate,
    it applies the yld_age_restrictions. If affected measure is
    excess_mortality_rate, it applies the yll_age_restrictions to filter
    the relative_risk data"""

    temp = []
    affected_entities = set(data.affected_entity)
    affected_measures = set(data.affected_measure)
    for cause, measure in product(affected_entities, affected_measures):
        df = data[(data.affected_entity == cause) & (data.affected_measure == measure)]
        cause = get_gbd_2020_entity(EntityKey(f'cause.{cause}.{measure}'))
        if measure == 'excess_mortality_rate':
            start, end = vi_utils.get_age_group_ids_by_restriction(cause, 'yll')
        else:  # incidence_rate
            start, end = vi_utils.get_age_group_ids_by_restriction(cause, 'yld')
        temp.append(df[df.age_group_id.isin(range(start, end + 1))])
    data = pd.concat(temp)
    return data
