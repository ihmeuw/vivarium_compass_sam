#############
# Scenarios #
#############

class Scenario:

    def __init__(self, has_wasting_prevention: bool):
        self.has_sqlns = has_wasting_prevention


SCENARIOS = {
    'baseline': Scenario(False),
    'sqlns': Scenario(True),
}
