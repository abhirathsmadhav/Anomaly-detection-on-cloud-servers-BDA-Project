import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

print("1. Reading outputs...")
events_df = pd.read_csv('outputs/final_ann_event_results.csv')
interval_df = pd.read_parquet('outputs/final_ann_interval_results.parquet')

with open('outputs/final_ann_results.json', 'r') as f:
    ann_results = json.load(f)

# Find the threshold for MSE from the JSON
threshold = None
for res in ann_results:
    if 'MSE' in res['Threshold_Rule']:
        # We need the threshold value. The JSON doesn't store the exact threshold float for MSE 
        # because the field is "Threshold_Rule": "MSE >= Train_99.9th_Perc".
        pass

# We can re-derive the threshold from the predictions since it's a fixed threshold
# The minimum MSE where pred_anomaly is 1 or max MSE where pred_anomaly is 0
if 'pred_anomaly' in interval_df.columns:
    anomalous = interval_df[interval_df['pred_anomaly'] == 1]
    if not anomalous.empty:
        threshold = anomalous['mse'].min()
    else:
        threshold = 0.007075717225952203 # Fallback from previous logs
else:
    threshold = 0.007075717225952203 # Fallback

print("2. Identifying correctly detected anomaly windows...")
detected_events = events_df[events_df['Detected'] == True].copy()
if detected_events.empty:
    print("No events detected!")
    exit(1)

print("3. Selecting the BEST demonstration event...")
# We prioritize the highest peak MSE for a clear visual spike
best_event = detected_events.sort_values(by='Peak_anomaly_score_mse', ascending=False).iloc[0]

event_id = best_event['Anomaly_ID']
event_start = pd.to_datetime(best_event['Start'])
event_end = pd.to_datetime(best_event['End'])

print(f"\nSelected Event: {event_id}")
print(f"Start: {event_start}")
print(f"End: {event_end}")
print(f"Detected buckets: {best_event['Number_of_detected_buckets']}")
print(f"Peak MSE: {best_event['Peak_anomaly_score_mse']:.6f}")

# Extract a timeline around the event (e.g., -6 hours and +6 hours)
view_start = event_start - pd.Timedelta(hours=6)
view_end = event_end + pd.Timedelta(hours=6)

interval_df['timestamp'] = pd.to_datetime(interval_df['interval_start'], unit='s', utc=True)
demo_df = interval_df[(interval_df['timestamp'] >= view_start) & (interval_df['timestamp'] <= view_end)].copy()

demo_df['anomaly_threshold'] = threshold
# recreate ground_truth column based on overlap
demo_intervals = demo_df['interval_start'].values
e_start_ts = (event_start - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')
e_end_ts = (event_end - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')

if e_start_ts == e_end_ts:
    mask = (demo_intervals <= e_start_ts) & (e_start_ts < demo_intervals + 300)
else:
    mask = np.maximum(demo_intervals, e_start_ts) < np.minimum(demo_intervals + 300, e_end_ts)

demo_df['ground_truth'] = mask.astype(int)

# Create the clean CSV
demo_out = demo_df[['timestamp', 'mse', 'anomaly_threshold', 'pred_anomaly', 'ground_truth']].copy()
demo_out.rename(columns={'timestamp': 'interval_time', 'mse': 'reconstruction_error', 'pred_anomaly': 'predicted_anomaly'}, inplace=True)
demo_out.to_csv('outputs/demo_selected_anomaly.csv', index=False)
print("Saved outputs/demo_selected_anomaly.csv")

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(demo_out['interval_time'], demo_out['reconstruction_error'], color='blue', linewidth=1.5, label='Reconstruction Error (MSE)')
plt.axhline(threshold, color='black', linestyle='--', linewidth=2, label='Detection Threshold (99.9th %ile)')

# Highlight ground truth
plt.axvspan(event_start, event_end, color='red', alpha=0.2, label='Documented Anomaly Window')

# Highlight predicted anomalies
anoms = demo_out[demo_out['predicted_anomaly'] == 1]
plt.scatter(anoms['interval_time'], anoms['reconstruction_error'], color='red', s=40, zorder=5, label='Predicted Anomaly')

plt.title(f"Anomaly Demonstration: Event {event_id}", fontsize=14)
plt.xlabel("Time (UTC)", fontsize=12)
plt.ylabel("Reconstruction Error", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/demo_selected_anomaly.png', dpi=150)
print("Saved outputs/demo_selected_anomaly.png")

# Generate summary
summary_text = f"""FINAL DEMONSTRATION SUMMARY

Selected Event: {event_id}
Documented Start (UTC): {event_start}
Documented End (UTC): {event_end}

DETECTION STATISTICS
- Overlapping Test Intervals (Total buckets inside window): {mask.sum()}
- Intervals Flagged as Anomalous by Detector: {best_event['Number_of_detected_buckets']}
- Maximum Reconstruction Error (MSE): {best_event['Peak_anomaly_score_mse']:.5f}
- Final Decision Threshold: {threshold:.5f}

DEMO SUITABILITY
This event is highly suitable for a video demonstration because it contains a massive, unambiguous spike in reconstruction error that clearly breaches the established 99.9th percentile threshold. The timeline visualization provides a very clean, intuitive picture of the model recognizing abnormal 5XX error behavior.

DISPLAY RECOMMENDATION
For the live or recorded demo, display the exact timeline between:
{view_start.strftime('%Y-%m-%d %H:%M UTC')} and {view_end.strftime('%Y-%m-%d %H:%M UTC')}

This spans 6 hours before and after the event, offering viewers clear context of the stable baseline right before the massive system deviation occurs.
"""

with open('outputs/demo_summary.txt', 'w') as f:
    f.write(summary_text)

print("Saved outputs/demo_summary.txt")
print(summary_text)
