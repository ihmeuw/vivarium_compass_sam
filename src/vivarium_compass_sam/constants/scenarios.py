#############
# Scenarios #
#############

class Scenario:

    def __init__(self, has_alternative_treatment: bool):
        self.has_alternative_treatment = has_alternative_treatment


SCENARIOS = {
    'baseline': Scenario(False),
    'wasting_treatment': Scenario(True),
}
