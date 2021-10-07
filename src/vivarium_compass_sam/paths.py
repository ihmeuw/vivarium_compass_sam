from pathlib import Path

import vivarium_compass_sam
from vivarium_compass_sam.constants import metadata

BASE_DIR = Path(vivarium_compass_sam.__file__).resolve().parent

ARTIFACT_ROOT = Path(f"/share/costeffectiveness/artifacts/{metadata.PROJECT_NAME}/")
MODEL_SPEC_DIR = BASE_DIR / 'model_specifications'
RESULTS_ROOT = Path(f'/share/costeffectiveness/results/{metadata.PROJECT_NAME}/')
