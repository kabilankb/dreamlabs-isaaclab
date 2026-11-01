#!/usr/bin/env bash
# Add the PutAppleInBowl-v0 task (Dream's MuJoCo scene) to a YAMLab checkout. Additive and safe to run again:
# it copies three new files and adds two registrations, and never edits anything else.
#
#   yamlab6-task/apply.sh [~/yamlab6]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
yamlab=$(realpath -- "${1:-$HOME/yamlab6}")
[[ -d $yamlab/yamlab/envs/tasks ]] || { echo "not a YAMLab checkout: $yamlab" >&2; exit 1; }

install -m 644 "$here/put_apple_in_bowl.yaml" "$yamlab/yamlab/configs/tasks/put_apple_in_bowl.yaml"
install -m 644 "$here/put_apple_in_bowl_manager_cfg.py" "$here/put_apple_in_bowl_manager.py" "$yamlab/yamlab/envs/tasks/"

init=$yamlab/yamlab/envs/tasks/__init__.py
grep -q "put_apple_in_bowl_manager_cfg" "$init" || printf '%s\n' \
    "from .put_apple_in_bowl_manager_cfg import *" "from .put_apple_in_bowl_manager import *" >> "$init"

registry=$yamlab/yamlab/envs/robot_registry.py
grep -q '"PutAppleInBowl-v0"' "$registry" || sed -i 's|^\(\s*\)"HangMugOnTree-Mimic-v0": "yam",|&\n\1"PutAppleInBowl-v0": "yam",|' "$registry"
grep -q '"PutAppleInBowl-v0"' "$registry" || { echo "could not register the task in $registry; add \"PutAppleInBowl-v0\": \"yam\" to TASK_ROBOTS" >&2; exit 1; }
echo "PutAppleInBowl-v0 is in $yamlab"
