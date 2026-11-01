"""Check a LeRobot dataset against the bimanual reBot embodiment, locally: nothing is sent and no key is read.

    ~/dream-env/bin/python -m dreamlabs_isaaclab.validate DATASET_DIR
"""
import json
import sys
from pathlib import Path

import dream

config = json.loads((Path(__file__).parent / "configs/rebot_bimanual_embodiment.json").read_text())
report = dream.DreamClient().validate_dataset(sys.argv[1], config)
print(report)
sys.exit(0 if report.ok else 1)
