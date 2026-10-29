#!/usr/bin/env bash
# Launch Capitalisman with the TimesFM GPU environment and tuned defaults.
# Falls back to the plain app venv when the ML environment is not installed.
# All CAPITALISMAN_TIMESFM_* values can be overridden per-launch, e.g.:
#   CAPITALISMAN_TIMESFM_MAX_CONTEXT=1024 ./run-timesfm.sh
# Note: in the UI only CAPITALISMAN_TIMESFM_DEVICE is honored: its value
# (auto, cpu or cuda) preselects the TimesFM Device box. Context, batch size
# and the rest come from the Runtime Profile picker and detected hardware, and
# MODEL_ID is ignored because the UI always names a backend. The other values
# apply only where the runtime is built from the environment
# (TimesFMRuntimeConfig.from_environment).
set -euo pipefail
cd "$(dirname "$0")"

TIMESFM_ENV="${TIMESFM_ENV:-$HOME/.venvs/capitalisman-timesfm}"

if [ -x "$TIMESFM_ENV/bin/streamlit" ]; then
    export CAPITALISMAN_TIMESFM_MAX_CONTEXT="${CAPITALISMAN_TIMESFM_MAX_CONTEXT:-8192}"
    # forecast_context defaults to 1024 upstream and effective context is
    # min(forecast_context, max_context) — follow MAX_CONTEXT unless overridden.
    export CAPITALISMAN_TIMESFM_FORECAST_CONTEXT="${CAPITALISMAN_TIMESFM_FORECAST_CONTEXT:-$CAPITALISMAN_TIMESFM_MAX_CONTEXT}"
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
