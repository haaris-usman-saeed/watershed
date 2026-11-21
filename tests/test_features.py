import numpy as np
import pandas as pd
from src.features import construct_features

def test_target_and_past_weather():
    dates = pd.date_range("2020-01-01", periods=160, freq="D")
    df = pd.DataFrame({"date": dates, "station_id": "X", "reservoir_name": "Test", "capacity_af":1000,
                       "storage_af":np.arange(160) * 2 + 300,
                       "rain_mm":1., "temp_c":20., "humidity_pct":50.})
    result = construct_features(df)
    assert len(result) > 0
    assert np.isclose(result.target_delta_pp.iloc[0], 6.0)
    assert np.isclose(result.rain_30d_mm.iloc[0], 30.0)
    assert all(result.target_date > result.date)

if __name__ == "__main__":
    test_target_and_past_weather(); print("feature smoke test passed")
