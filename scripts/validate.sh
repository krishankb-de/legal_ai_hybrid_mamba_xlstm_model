#!/usr/bin/env bash
# validate.sh -- the local gate (plan §8.3). A code box is ticked only after this exits 0.
#
#   bash scripts/validate.sh            # every code box: fast selection (no slow/cuda/multigpu/network)
#   bash scripts/validate.sh --full     # every Pn-Z gate: adds the slow tests
#   bash scripts/validate.sh --ci       # what the GitHub Actions `test` job runs: --full + coverage
#                                       # floor + JUnit output, shellcheck mandatory
#   bash scripts/validate.sh --full --coverage   # --full plus a coverage report (P1-W5)
#
# Gates:
#   0  plan/state consistent        python3 scripts/plan_state.py check
#   1  environment == lock          uv lock --check; scripts/check_env.py
#   2  static                       ruff check; ruff format --check; bash -n and shellcheck on *.sh
#   3  model configs                scripts/check_configs.py (Hydra compose, operator pins, vocab)
#   4  tests                        pytest with the mode's marker selection
#   5  CPU smoke                    scripts/smoke_model.py (every mixer, every parameter gets a grad)
#   6  repository hygiene           scripts/check_repo_hygiene.py (.gitignore-aware)
#
# The interpreter is $PYTHON, else ./.venv/bin/python (built by `uv sync --locked`).
# LEXHYBRID_EXPECT_PYTHON=3.12 lets the CI forward-compat matrix entry pass gate 1.

set -uo pipefail   # -e deliberately omitted: every gate runs and records its own status

MODE=fast
COVERAGE=0
for arg in "$@"; do
  case "$arg" in
    --full) MODE=full ;;
    --ci) MODE=ci; COVERAGE=1 ;;
    --coverage) COVERAGE=1 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "No venv interpreter at $PYTHON. Build it with: uv sync --locked" >&2
  exit 1
fi
UV="${UV:-$(command -v uv || true)}"

MARKERS_FAST="not slow and not cuda and not multigpu and not network"
MARKERS_FULL="not cuda and not multigpu and not network"
if [[ "$MODE" == fast ]]; then MARKERS="$MARKERS_FAST"; else MARKERS="$MARKERS_FULL"; fi

if [[ -t 1 ]]; then RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; NC=$'\033[0m'; else RED=""; GREEN=""; YELLOW=""; NC=""; fi
declare -a PASSED=() FAILED=() WARNED=()
pass() { PASSED+=("$1"); echo "${GREEN}[PASS]${NC} $1"; }
fail() { FAILED+=("$1"); echo "${RED}[FAIL]${NC} $1"; }
warn() { WARNED+=("$1"); echo "${YELLOW}[WARN]${NC} $1"; }
run_gate() {  # name, command...
  local name="$1"; shift
  if "$@"; then pass "$name"; else fail "$name"; fi
}

echo "================================================================"
echo " validate.sh  mode=$MODE  coverage=$COVERAGE"
echo " repo:   $REPO_ROOT"
echo " python: $PYTHON ($("$PYTHON" -c 'import sys; print(sys.version.split()[0])'))"
echo "================================================================"

echo; echo "-- Gate 0: plan and state --"
run_gate "plan/state consistent" "$PYTHON" scripts/plan_state.py check

echo; echo "-- Gate 1: environment --"
if [[ -n "$UV" ]]; then
  run_gate "uv.lock matches pyproject.toml" "$UV" lock --check
else
  fail "uv not found on PATH (the environment is defined by uv.lock)"
fi
run_gate "installed environment == uv.lock" "$PYTHON" scripts/check_env.py

echo; echo "-- Gate 2: static --"
run_gate "ruff check" "$PYTHON" -m ruff check .
run_gate "ruff format --check" "$PYTHON" -m ruff format --check .
SHELL_SCRIPTS=()   # portable read loop: macOS ships bash 3.2, which has no mapfile
while IFS= read -r f; do SHELL_SCRIPTS+=("$f"); done < <(find scripts -name '*.sh' -type f | sort)
SYNTAX_OK=1
for f in "${SHELL_SCRIPTS[@]}"; do bash -n "$f" || { echo "  syntax error: $f"; SYNTAX_OK=0; }; done
if [[ $SYNTAX_OK -eq 1 ]]; then pass "bash -n on ${#SHELL_SCRIPTS[@]} shell scripts"; else fail "bash -n"; fi
if command -v shellcheck >/dev/null 2>&1; then
  run_gate "shellcheck" shellcheck -x "${SHELL_SCRIPTS[@]}"
elif [[ "$MODE" == ci ]]; then
  fail "shellcheck not installed (required in --ci)"
else
  warn "shellcheck not installed locally; CI runs it"
fi

echo; echo "-- Gate 3: model configs --"
run_gate "every model yaml composes and pins its operators" "$PYTHON" scripts/check_configs.py

echo; echo "-- Gate 4: tests ($MARKERS) --"
PYTEST_ARGS=(-m "$MARKERS" --tb=short -q)
if [[ $COVERAGE -eq 1 ]]; then PYTEST_ARGS+=(--cov=lexhybrid --cov-report=term --cov-report=xml); fi
if [[ "$MODE" == ci ]]; then PYTEST_ARGS+=(--junitxml=junit.xml); fi
run_gate "pytest" "$PYTHON" -m pytest tests/ "${PYTEST_ARGS[@]}"

echo; echo "-- Gate 5: CPU smoke over every mixer --"
run_gate "forward/backward, every parameter gets a gradient" "$PYTHON" scripts/smoke_model.py

echo; echo "-- Gate 6: repository hygiene --"
# .gitignore-aware in every mode: by now the venv and caches exist. The CI `hygiene` job runs
# `check_repo_hygiene.py --ci` on a pristine checkout instead, where the tree IS the pushed tree.
run_gate "working tree (minus .gitignore) is clean" "$PYTHON" scripts/check_repo_hygiene.py

echo
echo "========================= SUMMARY ========================="
for p in "${PASSED[@]+"${PASSED[@]}"}"; do echo "  ${GREEN}[PASS]${NC} $p"; done
for w in "${WARNED[@]+"${WARNED[@]}"}"; do echo "  ${YELLOW}[WARN]${NC} $w"; done
for f in "${FAILED[@]+"${FAILED[@]}"}"; do echo "  ${RED}[FAIL]${NC} $f"; done
echo "==========================================================="
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "${RED}${#FAILED[@]} gate(s) failed.${NC}"
  exit 1
fi
echo "${GREEN}All gates passed (mode=$MODE).${NC}"
exit 0
