from pathlib import Path
import json
import pandas as pd
import streamlit as st
ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="WatershedShift", page_icon="💧", layout="wide")
st.title("WatershedShift | Reservoir storage forecasting")
st.caption("Research prototype · historical held-out forecasts · not an operational water-safety warning system")
metrics_file = ROOT / "outputs/metrics.json"
pred_file = ROOT / "outputs/test_predictions.csv"
if not (metrics_file.exists() and pred_file.exists()):
    st.warning("Run data ingestion, feature construction and training first. See README.md.")
    st.stop()
metrics = json.loads(metrics_file.read_text())
pred = pd.read_csv(pred_file, parse_dates=["date", "target_date"])
st.subheader("Evaluation on unseen 2024–2025 dates")
cols = st.columns(3)
cols[0].metric("ML MAE", f"{metrics['ML']:.2f} pts")
cols[1].metric("Best simple baseline MAE", f"{min(metrics['Persistence'], metrics['30-day trend'], metrics['Historical seasonal']):.2f} pts")
cols[2].metric("90% interval coverage", f"{metrics['interval_90_coverage']:.1%}")
st.write("Errors are measured in percentage points of the reservoir's nominal storage capacity, not percent of current storage.")
station = st.selectbox("Reservoir", sorted(pred.station_id.unique()))
sub = pred[pred.station_id == station].sort_values("date").copy()
sub = sub.set_index("date")
st.line_chart(sub[["target_delta_pp", "pred_delta_pp", "baseline_trend_pp"]].rename(columns={"target_delta_pp":"Observed 30-day storage change", "pred_delta_pp":"ML predicted", "baseline_trend_pp":"Trend baseline"}))
st.caption("Historical out-of-sample forecasts only. Prediction intervals may not achieve nominal coverage on future data.")
st.dataframe(sub[["reservoir_name", "target_date", "target_delta_pp", "pred_delta_pp", "lower_90_pp", "upper_90_pp"]].tail(20))
st.download_button("Download historical predictions", pred.to_csv(index=False).encode(), "historical_forecasts.csv", "text/csv")
