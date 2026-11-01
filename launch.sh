#!/usr/bin/env bash
# Install what dreamlabs-isaaclab needs, then start its launcher page. Safe to run again: every step checks first.
#
#   ./launch.sh                 install what is missing, then open the launcher
#   ./launch.sh --check         only report what is ready and what is missing; changes nothing (exit 1 if not ready)
#   ./launch.sh --install-only  install, but do not start the launcher
#   ./launch.sh --no-browser    print the launcher address instead of opening it
#   ./launch.sh --port 9000     start on another port (the next free one if taken)
#
# What it does, in order:
#   1. Isaac Sim's Python          required; it only checks (Isaac Sim and Isaac Lab are installed as in the course)
#   2. the Dream environment       ~/dream-env: creates it with uv and installs `dreamlabs`, `pyarrow`, `av`
#   3. your Dream API key          only checks that one is saved; add or replace it on the launcher page
#   4. the MuJoCo-scene assets     converts Dream's apple/bowl/... to USD in ~/.dreamlabs-isaaclab/assets
#   5. the YAMLab task             adds PutAppleInBowl-v0 to ~/yamlab6 (three new files, two registrations) if it is there
#   6. the reBot apple task        found in your rebot-duo-course checkout (optional: only the reBot apple robot needs it)
#   7. the launcher page           stdlib only; listens on 127.0.0.1
# It never pipes a downloaded script into a shell and never prints, logs or passes on your API key.
#
# Overrides (environment): DREAMLABS_ISAAC_PYTHON, DREAMLABS_DREAM_ENV, DREAMLABS_YAMLAB, DREAMLABS_ASSETS_DIR,
# DREAMLABS_COURSE_REPO (the rebot-duo-course checkout that has the reBot apple task).
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
isaac_python=${DREAMLABS_ISAAC_PYTHON:-$HOME/miniconda3/envs/env_yamlab6/bin/python}
dream_env=${DREAMLABS_DREAM_ENV:-$HOME/dream-env}
dream_python=$dream_env/bin/python
yamlab=${DREAMLABS_YAMLAB:-$HOME/yamlab6}
assets=${DREAMLABS_ASSETS_DIR:-$HOME/.dreamlabs-isaaclab/assets/yam_mujoco_scene}
credentials=$HOME/.dreamlabs/credentials

check_only=0 install_only=0 launcher_args=()
while (($#)); do
    case $1 in
        --check) check_only=1 ;;
        --install-only) install_only=1 ;;
        --no-browser) launcher_args+=(--no-browser) ;;
        --port) shift; [[ ${1:-} =~ ^[0-9]+$ ]] || { echo "--port needs a number" >&2; exit 2; }; launcher_args+=(--port "$1") ;;
        -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

problems=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
todo() { printf '  \033[33m•\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*"; problems=$((problems + 1)); }
step() { printf '\n%s\n' "$*"; }
doing() { if ((check_only)); then todo "would: $*"; problems=$((problems + 1)); return 1; fi; printf '    … %s\n' "$*"; }

# 0. the Python that runs this launcher page (the standard library is enough)
step "Launcher"
if python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then ok "python3 $(python3 -c 'import platform; print(platform.python_version())')"
else bad "python3 3.10 or newer is needed to run the launcher page"; fi

# 1. Isaac Sim's Python
step "1. Isaac Sim and Isaac Lab"
if [[ -x $isaac_python ]] && "$isaac_python" -c 'import isaacsim, isaaclab' 2>/dev/null; then ok "$isaac_python"
elif [[ -x $isaac_python ]]; then bad "$isaac_python has no isaacsim / isaaclab: install them as in the course, Module 0"
else bad "no Isaac Sim Python at $isaac_python (set DREAMLABS_ISAAC_PYTHON, or install it as in the course, Module 0)"; fi

# Without Isaac Sim there is nothing to launch: stop here, before installing anything. (--check goes on and reports all.)
if ((problems && !check_only)); then
    echo >&2
    echo "Isaac Sim and Isaac Lab are required first; nothing was installed. See the course, Module 0." >&2
    exit 1
fi

# 2. the Dream environment
step "2. Dream environment ($dream_env)"
dream_ready() { [[ -x $dream_python ]] && "$dream_python" -c 'import dream, robots, pyarrow, av' 2>/dev/null; }
if dream_ready; then ok "dreamlabs $("$dream_python" -m pip show dreamlabs 2>/dev/null | sed -n 's/^Version: //p')"
elif doing "create it and install dreamlabs, pyarrow, av"; then
    if [[ ! -x $dream_python ]]; then
        if command -v uv >/dev/null; then uv venv --seed --python 3.12 --allow-existing "$dream_env"
        else
            py=""; for v in 3.12 3.13 3.11; do command -v "python$v" >/dev/null && { py=python$v; break; }; done
            [[ -n $py ]] || { bad "needs uv (https://docs.astral.sh/uv/) or Python 3.11 to 3.13 to make the Dream environment"; py=; }
            [[ -z $py ]] || "$py" -m venv "$dream_env"
        fi
    fi
    if [[ -x $dream_python ]]; then
        "$dream_python" -m pip install --quiet --upgrade pip dreamlabs pyarrow av
        dream_ready && ok "installed" || bad "the Dream environment is still missing dreamlabs / pyarrow / av"
    fi
fi

# 3. the API key
step "3. Dream API key"
if grep -q '^DREAM_API_KEY=.' "$credentials" 2>/dev/null; then ok "saved in $credentials (checked on the launcher page)"
else todo "none saved yet: add it on the launcher page (step 2 there). It is verified with Dream, then stored mode 600."; fi

# 4. the converted MuJoCo-scene assets
step "4. Dream's MuJoCo-scene assets"
assets_ready() { for d in Apple/apple_000 Banana/banana_000 Bowl/bowl_000 Orange/orange_000 WaterBottle/water_bottle_000 Plate/plate_000; do
    [[ -f $assets/$d/${d##*/}.usd ]] || return 1; done; }
if assets_ready; then ok "$assets"
elif [[ ! -x $isaac_python ]] || ! dream_ready; then todo "needs steps 1 and 2 first"
elif doing "convert them with Isaac's Python from the Dream example"; then
    source_dir=$(compgen -G "$dream_env/lib/python3.*/site-packages/dream_examples/yam_mujoco/assets" | head -1 || true)
    if [[ -z $source_dir ]]; then bad "Dream's yam_mujoco example is not in $dream_env (pip install dreamlabs)"
    else
        PYTHONPATH=$here/src "$isaac_python" -I -c "import sys; sys.path.insert(0, '$here/src'); from dreamlabs_isaaclab.mujoco_scene_assets import main; sys.exit(main(['--out', '$assets', '--source', '$source_dir']))" >/dev/null
        assets_ready && ok "converted into $assets" || bad "the conversion did not produce all six objects"
    fi
fi

# 5. the YAMLab task
step "5. YAMLab apple-in-bowl task ($yamlab)"
yamlab_ready() { [[ -f $yamlab/yamlab/envs/tasks/put_apple_in_bowl_manager.py ]] && grep -q put_apple_in_bowl "$yamlab/yamlab/envs/tasks/__init__.py"; }
if [[ ! -d $yamlab/yamlab ]]; then todo "no YAMLab checkout here: the YAM robots will show as not ready (set DREAMLABS_YAMLAB)"
elif yamlab_ready; then ok "PutAppleInBowl-v0 is in $yamlab"
elif doing "add it (three new files, two registrations; nothing else is edited)"; then
    "$here/yamlab6-task/apply.sh" "$yamlab" >/dev/null
    yamlab_ready && ok "added" || bad "could not add the task to $yamlab"
fi

step "6. reBot apple task (needs the rebot-duo-course repository)"
course=""
for candidate in "${DREAMLABS_COURSE_REPO:-}" "$here/.." "$HOME/rebot-duo-course"; do
    if [[ -n $candidate && -f $candidate/rebot-sim-pipeline/src/rebot_sim_pipeline/lab/configs/tasks/apple_in_bowl.yaml ]]; then
        course=$(cd "$candidate" && pwd); break
    fi
done
if [[ -n $course ]]; then ok "rebot-sim-pipeline with the apple task: $course"; export DREAMLABS_COURSE_REPO=$course
else todo "not found: the reBot apple robot will show as not ready. git clone https://github.com/kabilankb/rebot-duo-course and set DREAMLABS_COURSE_REPO to it."; fi

echo
if ((check_only)); then
    ((problems == 0)) && echo "Everything is ready." || echo "$problems thing(s) to do: run ./launch.sh to install them."
    ((problems == 0)); exit
fi
if ((problems)); then echo "Not starting: $problems problem(s) above." >&2; exit 1; fi
if ((install_only)); then echo "Installed. Start the launcher with ./launch.sh"; exit 0; fi
echo "Starting the launcher (Ctrl-C stops it and any run it started)."
PYTHONPATH=$here/src exec python3 -u -m dreamlabs_isaaclab.launcher "${launcher_args[@]}"
