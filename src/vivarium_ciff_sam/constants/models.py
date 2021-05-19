from vivarium_ciff_sam.constants import data_keys


class TransitionString(str):

    def __new__(cls, value):
        # noinspection PyArgumentList
        obj = str.__new__(cls, value.lower())
        obj.from_state, obj.to_state = value.split('_TO_')
        return obj


###########################
# Disease Model variables #
###########################

# TODO input details of model states and transitions
DIARRHEA_MODEL_NAME = data_keys.DIARRHEA.name
DIARRHEA_SUSCEPTIBLE_STATE_NAME = f'susceptible_to_{DIARRHEA_MODEL_NAME}'
DIARRHEA_STATE_NAME = DIARRHEA_MODEL_NAME
DIARRHEA_MODEL_STATES = (DIARRHEA_SUSCEPTIBLE_STATE_NAME, DIARRHEA_STATE_NAME)
DIARRHEA_MODEL_TRANSITIONS = (
    TransitionString(f'{DIARRHEA_SUSCEPTIBLE_STATE_NAME}_TO_{DIARRHEA_STATE_NAME}'),
    TransitionString(f'{DIARRHEA_STATE_NAME}_TO_{DIARRHEA_SUSCEPTIBLE_STATE_NAME}'),
)
MEASLES_MODEL_NAME = data_keys.MEASLES.name
MEASLES_SUSCEPTIBLE_STATE_NAME = f'susceptible_to_{MEASLES_MODEL_NAME}'
MEASLES_STATE_NAME = MEASLES_MODEL_NAME
MEASLES_MODEL_STATES = (MEASLES_SUSCEPTIBLE_STATE_NAME, MEASLES_STATE_NAME)
MEASLES_MODEL_TRANSITIONS = (
    TransitionString(f'{MEASLES_SUSCEPTIBLE_STATE_NAME}_TO_{MEASLES_STATE_NAME}'),
    TransitionString(f'{MEASLES_STATE_NAME}_TO_{MEASLES_SUSCEPTIBLE_STATE_NAME}'),
)
LRI_MODEL_NAME = data_keys.LRI.name
LRI_SUSCEPTIBLE_STATE_NAME = f'susceptible_to_{LRI_MODEL_NAME}'
LRI_STATE_NAME = LRI_MODEL_NAME
LRI_MODEL_STATES = (LRI_SUSCEPTIBLE_STATE_NAME, LRI_STATE_NAME)
LRI_MODEL_TRANSITIONS = (
    TransitionString(f'{LRI_SUSCEPTIBLE_STATE_NAME}_TO_{LRI_STATE_NAME}'),
    TransitionString(f'{LRI_STATE_NAME}_TO_{LRI_SUSCEPTIBLE_STATE_NAME}'),
)

STATE_MACHINE_MAP = {
    DIARRHEA_MODEL_NAME: {
        'states': DIARRHEA_MODEL_STATES,
        'transitions': DIARRHEA_MODEL_TRANSITIONS,
    },
    MEASLES_MODEL_NAME: {
        'states': MEASLES_MODEL_STATES,
        'transitions': MEASLES_MODEL_TRANSITIONS,
    },
    LRI_MODEL_NAME: {
        'states': LRI_MODEL_STATES,
        'transitions': LRI_MODEL_TRANSITIONS,
    },
}


STATES = tuple(state for model in STATE_MACHINE_MAP.values() for state in model['states'])
TRANSITIONS = tuple(state for model in STATE_MACHINE_MAP.values() for state in model['transitions'])
