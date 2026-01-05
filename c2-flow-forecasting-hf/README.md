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

### 📊 Input Data

The system operates on aggregated network flow data, derived from raw NetFlow / VPC Flow Logs / firewall logs.

Raw flow schema (example)
timestamp, src_ip, dst_ip, src_port, dst_port, protocol, bytes, packets, duration, flags

Aggregated features per time window

Examples include:
- flow_count
- bytes_sum, bytes_mean, bytes_std
- packets_sum
- avg_duration, duration_std
- unique_dst_ports
- tcp_ratio, udp_ratio
- port_443_ratio
- dns_ratio
- small_flow_ratio

These features are intentionally generic and vendor-agnostic.

## 🧠 Model Training Strategy

- Training data: First N hours of traffic (baseline)
- Scoring data: Entire dataset (including potential attacks)
- Windowing:
  - Context length: historical window used for forecasting
  - Prediction length: future window to evaluate
- Scaling: StandardScaler fit on baseline only
- Lag safety: Extra history added to support HF lag extraction

## 🚨 Anomaly Scoring & Alerting
Anomaly score
For each future window:
score = mean( |observed − forecast_mean| / forecast_std )

This is a normalized residual score across all features.

### Persistence logic

A window is only flagged if:
- Score exceeds a threshold AND
- The condition persists across multiple consecutive windows

This is critical for detecting low-and-slow beaconing while suppressing one-off spikes.

## 📈 Outputs

The pipeline produces:

📄 CSV files
- results/top_scores.csv – highest anomaly scores
- results/alerts.csv – all alerts after persistence logic
- results/c2_alerts.csv – alerts involving a specific C2 destination

📊 Visualization
- results/anomaly_scores.png – anomaly score over time for top pairs and C2 traffic

These artifacts make the results SOC-reviewable and demo-ready.

## ▶️ How to Run
1️⃣ Create and activate virtual environment (Windows)
py -m venv .venv
.venv\Scripts\Activate.ps1

2️⃣ Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

3️⃣ Run the pipeline
python -m src.main `
  --csv data/flows_test.csv `
  --window 5min `
  --train_hours 12 `
  --anomaly_threshold 0.52 `
  --persistence_windows 4

4️⃣ Check outputs
results/
├── top_scores.csv
├── alerts.csv
├── c2_alerts.csv
└── anomaly_scores.png

### 🧪 Example Use Case Demonstrated

The synthetic dataset includes a low-and-slow HTTPS beacon:

10.0.2.33 → 198.51.100.77

Characteristics:
- Small payloads
- Periodic connections
- Port 443
- Long dwell time

The model successfully flags this behavior without signatures or IOCs.

## 🚀 Why This Matters

This project demonstrates how modern ML + time-series forecasting can:
- Detect stealthy C2 activity earlier
- Reduce reliance on brittle rules
- Generalize across environments
- Scale to cloud and enterprise networks

It reflects how real SOC detection pipelines increasingly combine:
- ML forecasting
- Statistical scoring
- Persistence-based alerting
- Analyst-friendly outputs

## 📌 Future Enhancements
- Beacon-likeness heuristics (variance, periodicity)
- ASN / geo enrichment
- Per-host baselining
- Online / streaming inference
- LLM-generated alert explanations
