## 🕵️‍♂️ Early Detection of Low-and-Slow C2 via Network Flow Time-Series Forecasting
Overview

This project demonstrates an early-detection framework for low-and-slow Command-and-Control (C2) activity using network flow time-series forecasting powered by the Hugging Face TimeSeriesTransformer.

Instead of relying on signatures, indicators of compromise (IOCs), or identity telemetry, this system models expected network behavior over time and flags persistent deviations that are characteristic of stealthy beaconing malware.

The result is a behavioral, model-driven detection pipeline suitable for modern SOC environments where attackers deliberately blend into “normal” traffic.

## 🎯 Problem Statement

Traditional C2 detection struggles with low-and-slow malware because:
- Beacon intervals are long and irregular
- Traffic volumes are intentionally small
- Destinations often use HTTPS (port 443) and cloud infrastructure
- Signatures and thresholds fail to trigger

Static rules like:
- “bytes > X”
- “connections per minute > Y”
- “known bad IPs”

either miss the attack or generate excessive false positives.

## 💡 Solution Approach

We reframe the problem as a time-series forecasting task:

“Given historical network flow behavior for a (src_ip → dst_ip) pair, how surprising is the future behavior?”

High-level idea

1. Aggregate raw NetFlow records into fixed-width time windows
2. Train a probabilistic forecasting model on baseline (clean) traffic
3. Forecast expected future behavior
4. Score deviations between forecasted and observed behavior
5. Apply persistence logic to detect low-and-slow anomalies

This allows us to catch small but consistent deviations that would never trip static thresholds.

## 🤗 Why Hugging Face TimeSeriesTransformer?

We use Hugging Face’s TimeSeriesTransformerForPrediction because it provides:

# ✅ Probabilistic forecasting
- Generates multiple future samples, not a single point estimate
- Enables uncertainty-aware anomaly scoring

# ✅ Native temporal modeling
- Learns daily / weekly cycles automatically
- Uses lagged subsequences internally (ideal for network telemetry)

# ✅ Production-grade architecture
- Transformer encoder-decoder design
- Scales to multivariate time series
- Clean API for training and generation

# ✅ Security-relevant advantage

Unlike classical ARIMA or simple LSTMs, this model:
- Handles non-stationary traffic
- Captures subtle temporal drift
- Produces interpretable residual-based anomaly scores

## Input schema (CSV)
timestamp,src_ip,dst_ip,src_port,dst_port,protocol,bytes,packets,duration,flags

## Install
pip install -r requirements.txt

## Run
python -m src.main --csv data/flows_test.csv --window 5min

Optional tuning:
python -m src.main --csv src/data/flows_test.csv --window 5min --anomaly_threshold 2.5 --persistence_windows 3
