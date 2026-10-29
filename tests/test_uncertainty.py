import numpy as np
import pytest

from research.uncertainty import paired_mae_interval


def test_paired_interval_preserves_pairing_and_sign():
    errors = np.linspace(0, 1, 30)
    result = paired_mae_interval(errors, errors + 0.1, horizon=5, step=5)
    assert result["status"] == "estimated"
    assert result["lower"] == pytest.approx(0.1)
    assert result["upper"] == pytest.approx(0.1)
    assert result["multiplicity_adjusted"] is False


def test_overlap_and_small_samples_do_not_produce_spurious_intervals():
    result = paired_mae_interval(np.zeros(12), np.ones(12), horizon=10, step=1)
    assert result["block_length"] == 10
    assert result["status"] == "insufficient_observations"
    assert result["lower"] is result["upper"] is None
    assert paired_mae_interval([1], [2], horizon=1, step=1)["lower"] is None


def test_uncertainty_is_reproducible_and_rejects_unpaired_data():
    a = np.arange(40) / 100
    first = paired_mae_interval(a, a[::-1], horizon=5, step=5)
    assert first == paired_mae_interval(a, a[::-1], horizon=5, step=5)
    assert first["lower"] < 0 < first["upper"]
    with pytest.raises(ValueError, match="matching"):
        paired_mae_interval([1], [1, 2], horizon=1, step=1)
    with pytest.raises(ValueError, match="finite"):
        paired_mae_interval([np.nan], [1], horizon=1, step=1)
