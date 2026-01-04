# Early Detection of Low-and-Slow C2 via Network Flow Time-Series Forecasting (Hugging Face)

This project detects stealthy Command-and-Control (C2) beaconing patterns using only network flow logs
(no packet payloads, no decryption). It trains a Hugging Face TimeSeriesTransformer to forecast expected
traffic behavior per src->dst pair and flags persistent forecast deviations.

## Input schema (CSV)
timestamp,src_ip,dst_ip,src_port,dst_port,protocol,bytes,packets,duration,flags

## Install
pip install -r requirements.txt

## Run
python -m src.main --csv data/flows_test.csv --window 5min

Optional tuning:
python -m src.main --csv data/flows_test.csv --window 5min --anomaly_threshold 2.5 --persistence_windows 3
