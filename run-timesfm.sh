#!/usr/bin/env bash
# Launch Capitalisman with the TimesFM GPU environment and tuned defaults.
# Falls back to the plain app venv when the ML environment is not installed.
# All CAPITALISMAN_TIMESFM_* values can be overridden per-launch, e.g.:
#   CAPITALISMAN_TIMESFM_MAX_CONTEXT=1024 ./run-timesfm.sh
set -euo pipefail
cd "$(dirname "$0")"

TIMESFM_ENV="${TIMESFM_ENV:-$HOME/.venvs/capitalisman-timesfm}"

if [ -x "$TIMESFM_ENV/bin/streamlit" ]; then
    export CAPITALISMAN_TIMESFM_MAX_CONTEXT="${CAPITALISMAN_TIMESFM_MAX_CONTEXT:-8192}"
    export CAPITALISMAN_TIMESFM_BATCH_SIZE="${CAPITALISMAN_TIMESFM_BATCH_SIZE:-128}"
    export CAPITALISMAN_TIMESFM_DEVICE="${CAPITALISMAN_TIMESFM_DEVICE:-cuda}"
    export PYTHONNOUSERSITE=1
    echo "Launching with TimesFM (${TIMESFM_ENV})"
    echo "  context=${CAPITALISMAN_TIMESFM_MAX_CONTEXT} batch=${CAPITALISMAN_TIMESFM_BATCH_SIZE} device=${CAPITALISMAN_TIMESFM_DEVICE}"
    exec "$TIMESFM_ENV/bin/streamlit" run app.py "$@"
elif [ -x "venv/bin/streamlit" ]; then
    echo "TimesFM environment not found at ${TIMESFM_ENV} — launching without the"
    echo "AI forecaster (all other indicators work). See docs/TIMESFM.md to set it up."
    exec venv/bin/streamlit run app.py "$@"
else
    echo "No environment found. Create one first:" >&2
    echo "  python3 -m venv venv && venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi
