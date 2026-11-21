import pandas as pd

from src.compare_resopsus import AF_PER_MCM, match


def test_mcm_to_acre_feet_constant():
    assert abs(AF_PER_MCM - 810.7132) < 1e-3


def test_match_tolerates_rounding_and_conversion_constant_only():
    a = pd.Series([4_500_439.0, 1_000.0])
    assert match(a, pd.Series([4_500_432.0, 1_001.5])).all()        # 7 AF at 4.5M AF (1.5e-6), 1.5 AF rounding
    assert not match(pd.Series([1_000_000.0]), pd.Series([1_000_100.0])).any()
