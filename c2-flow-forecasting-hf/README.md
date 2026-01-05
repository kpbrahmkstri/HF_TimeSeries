##🕵️‍♂️ Early Detection of Low-and-Slow C2 via Network Flow Time-Series Forecasting
Overview

This project demonstrates an early-detection framework for low-and-slow Command-and-Control (C2) activity using network flow time-series forecasting powered by the Hugging Face TimeSeriesTransformer.

Instead of relying on signatures, indicators of compromise (IOCs), or identity telemetry, this system models expected network behavior over time and flags persistent deviations that are characteristic of stealthy beaconing malware.

The result is a behavioral, model-driven detection pipeline suitable for modern SOC environments where attackers deliberately blend into “normal” traffic.

## Input schema (CSV)
timestamp,src_ip,dst_ip,src_port,dst_port,protocol,bytes,packets,duration,flags

## Install
pip install -r requirements.txt

## Run
python -m src.main --csv data/flows_test.csv --window 5min

Optional tuning:
python -m src.main --csv src/data/flows_test.csv --window 5min --anomaly_threshold 2.5 --persistence_windows 3
