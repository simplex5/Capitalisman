"""Historical evaluations must not treat unfinished daily candles as closes."""

import pandas as pd
import pytest

from research.data import completed_daily_bars


@pytest.mark.parametrize("tz,last", [("UTC", "2026-09-26"), ("America/New_York", "2026-09-25"), (None, "2026-09-26")])
def test_completed_cutoff_uses_provider_timezone_and_excludes_future(tz, last):
    frame = pd.DataFrame({"Close": 100.}, index=pd.date_range("2026-09-24", periods=5, tz=tz))
    result = completed_daily_bars(frame, as_of="2026-09-27T01:00:00Z")
    assert str(result.index[-1].date()) == last
    assert len(frame) == 5


def test_rejects_ambiguous_naive_cutoff():
    frame = pd.DataFrame(index=pd.date_range("2026-01-01", periods=5))
    with pytest.raises(ValueError, match="timezone-aware"):
        completed_daily_bars(frame, as_of="2026-01-03")
