####################
# Project metadata #
####################

PROJECT_NAME = 'vivarium_compass_sam'
CLUSTER_PROJECT = 'proj_cost_effect'

CLUSTER_QUEUE = 'all.q'
MAKE_ARTIFACT_MEM = '10G'
MAKE_ARTIFACT_CPU = '1'
MAKE_ARTIFACT_RUNTIME = '3:00:00'
MAKE_ARTIFACT_SLEEP = 10

LOCATIONS = [
    'Ethiopia'
]

ARTIFACT_INDEX_COLUMNS = [
    'sex',
    'age_start',
    'age_end',
    'year_start',
    'year_end'
]

GBD_2019_ROUND_ID = 6
GBD_2020_ROUND_ID = 7

GBD_2020_AGE_GROUPS = [2, 3, 388, 389, 238, 34]
