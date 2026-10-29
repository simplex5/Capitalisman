"""Conservative completed-session cutoff for daily historical research."""

import pandas as pd


def completed_daily_bars(frame: pd.DataFrame, *, as_of=None) -> pd.DataFrame:
    """Exclude today's date and future dates in the provider's index timezone.

    This deliberately also omits today's stock bar after its exchange closes;
    accepting same-day bars requires an exchange calendar and finality contract.
    Date-only indices without a timezone use UTC. Live prediction may still use
    the latest provider bar; this helper is for historical evaluation only.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Daily research data requires a DatetimeIndex")
    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if pd.isna(now) or now.tzinfo is None:
        raise ValueError("The research cutoff must be a valid timezone-aware timestamp")
    cutoff = now.tz_convert(frame.index.tz or "UTC").normalize()
    if frame.index.tz is None:
        cutoff = cutoff.tz_localize(None)
    return frame.loc[frame.index.normalize() < cutoff].copy()
