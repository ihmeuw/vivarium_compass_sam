from pathlib import Path
from typing import Dict, NamedTuple, Union

import pandas as pd
import yaml

from vivarium_compass_sam.constants import results


SCENARIO_COLUMN = 'scenario'
GROUPBY_COLUMNS = [
    results.INPUT_DRAW_COLUMN,
    SCENARIO_COLUMN,
]
OUTPUT_COLUMN_SORT_ORDER = [
    'age_group',
    'sex',
    'year',
    'risk',
    'cause',
    'measure',
    'input_draw'
]


def make_measure_data(data):
    measure_data = MeasureData(
        population=get_population_data(data),
        ylls=get_by_cause_measure_data(data, 'ylls'),
        deaths=get_by_cause_measure_data(data, 'deaths'),
        disease_state_person_time=get_state_person_time_measure_data(data, 'disease_state_person_time'),
        disease_transition_count=get_transition_count_measure_data(data, 'disease_transition_count'),
        wasting_state_person_time=get_state_person_time_measure_data(data, 'wasting_state_person_time', False),
    )
    return measure_data


class MeasureData(NamedTuple):
    population: pd.DataFrame
    ylls: pd.DataFrame
    deaths: pd.DataFrame
    disease_state_person_time: pd.DataFrame
    disease_transition_count: pd.DataFrame
    wasting_state_person_time: pd.DataFrame

    def dump(self, output_dir: Path):
        for key, df in self._asdict().items():
            df.to_hdf(output_dir / f'{key}.hdf', key=key)
            df.to_csv(output_dir / f'{key}.csv')


def read_data(path: Path, single_run: bool) -> (pd.DataFrame, Dict[str, Union[str, int]]):
    data = pd.read_hdf(path)
    # noinspection PyUnresolvedReferences
    data = (data
            .reset_index(drop=True)
            .rename(columns={results.OUTPUT_SCENARIO_COLUMN: SCENARIO_COLUMN})
            )
    if single_run:
        data[results.INPUT_DRAW_COLUMN] = 0
        data[results.RANDOM_SEED_COLUMN] = 0
        data[SCENARIO_COLUMN] = 'baseline'
        keyspace = {results.INPUT_DRAW_COLUMN: [0],
                    results.RANDOM_SEED_COLUMN: [0],
                    results.OUTPUT_SCENARIO_COLUMN: ['baseline']}
    else:
        data[results.INPUT_DRAW_COLUMN] = data[results.INPUT_DRAW_COLUMN].astype(int)
        data[results.RANDOM_SEED_COLUMN] = data[results.RANDOM_SEED_COLUMN].astype(int)
        with (path.parent / 'keyspace.yaml').open() as f:
            keyspace = yaml.full_load(f)
    return data, keyspace


def filter_out_incomplete(data: pd.DataFrame, keyspace: Dict[str, Union[str, int]]):
    output = []
    for draw in keyspace[results.INPUT_DRAW_COLUMN]:
        # For each draw, gather all random seeds completed for all scenarios.
        random_seeds = set(keyspace[results.RANDOM_SEED_COLUMN])
        draw_data = data.loc[data[results.INPUT_DRAW_COLUMN] == draw]
        for scenario in keyspace[results.OUTPUT_SCENARIO_COLUMN]:
            seeds_in_data = draw_data.loc[
                data[SCENARIO_COLUMN] == scenario,
                results.RANDOM_SEED_COLUMN
            ].unique()
            random_seeds = random_seeds.intersection(seeds_in_data)
        draw_data = draw_data.loc[draw_data[results.RANDOM_SEED_COLUMN].isin(random_seeds)]
        output.append(draw_data)
    return pd.concat(output, ignore_index=True).reset_index(drop=True)


def aggregate_over_seed(data: pd.DataFrame) -> pd.DataFrame:
    non_count_columns = []
    for non_count_template in results.NON_COUNT_TEMPLATES:
        non_count_columns += results.RESULT_COLUMNS(non_count_template)
    count_columns = [c for c in data.columns if c not in non_count_columns + GROUPBY_COLUMNS]

    # non_count_data = data[non_count_columns + GROUPBY_COLUMNS].groupby(GROUPBY_COLUMNS).mean()
    count_data = data[count_columns + GROUPBY_COLUMNS].groupby(GROUPBY_COLUMNS).sum()
    return pd.concat([
        count_data,
        # non_count_data
    ], axis=1).reset_index()


def pivot_data(data: pd.DataFrame) -> pd.DataFrame:
    return (data
            .set_index(GROUPBY_COLUMNS)
            .stack()
            .reset_index()
            .rename(columns={f'level_{len(GROUPBY_COLUMNS)}': 'process', 0: 'value'}))


def sort_data(data: pd.DataFrame) -> pd.DataFrame:
    sort_order = [c for c in OUTPUT_COLUMN_SORT_ORDER if c in data.columns]
    other_cols = [c for c in data.columns if c not in sort_order]
    data = data[sort_order + other_cols].sort_values(sort_order)
    return data.reset_index(drop=True)


def split_processing_column(data: pd.DataFrame, has_wasting_stratification: bool = True) -> pd.DataFrame:
    if has_wasting_stratification:
        data['process'], data['wasting_state'] = data.process.str.split(f'_wasting_state_').str
    data['process'], data['age'] = data.process.str.split('_in_age_group_').str
    data['process'], data['sex'] = data.process.str.split('_among_').str
    data['year'] = data.process.str.split('_in_').str[-1]
    data['measure'] = data.process.str.split('_in_').str[:-1].apply(lambda x: '_in_'.join(x))
    return data.drop(columns='process')


def get_population_data(data: pd.DataFrame) -> pd.DataFrame:
    total_pop = pivot_data(data[[results.TOTAL_POPULATION_COLUMN]
                                + results.RESULT_COLUMNS('population')
                                + GROUPBY_COLUMNS])
    total_pop = total_pop.rename(columns={'process': 'measure'})
    return sort_data(total_pop)


def get_measure_data(data: pd.DataFrame, measure: str, has_wasting_stratification: bool = True) -> pd.DataFrame:
    data = pivot_data(data[results.RESULT_COLUMNS(measure) + GROUPBY_COLUMNS])
    data = split_processing_column(data, has_wasting_stratification)
    return sort_data(data)


def get_by_cause_measure_data(data: pd.DataFrame, measure: str,
                              has_wasting_stratification: bool = True) -> pd.DataFrame:
    data = get_measure_data(data, measure, has_wasting_stratification)
    data['measure'], data['cause'] = data.measure.str.split('_due_to_').str
    return sort_data(data)


def get_state_person_time_measure_data(data: pd.DataFrame, measure: str,
                                       has_wasting_stratification: bool = True) -> pd.DataFrame:
    data = get_measure_data(data, measure, has_wasting_stratification)
    data['measure'], data['cause'] = 'state_person_time', data.measure.str.split('_person_time').str[0]
    return sort_data(data)


def get_transition_count_measure_data(data: pd.DataFrame, measure: str,
                                      has_wasting_stratification: bool = True) -> pd.DataFrame:
    # Oops, edge case.
    data = data.drop(columns=[c for c in data.columns if 'event_count' in c and '2027' in c])
    data = get_measure_data(data, measure, has_wasting_stratification)
    return sort_data(data)
