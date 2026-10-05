import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Generating visualizations for Experiment 3...")
system_df = pd.read_parquet('outputs/system_df_experiment_3.parquet')
system_df['timestamp'] = pd.to_datetime(system_df['interval_start'], unit='s', utc=True)
df_anomalies = pd.read_csv('anomaly_windows.csv')

# Anomalous Context Fraction Over Time
plt.figure(figsize=(15, 5))
plt.plot(system_df['timestamp'], system_df['anomalous_context_fraction'], color='teal', alpha=0.8, label='Anomalous Context Fraction (Z > 5)')

# Highlight known anomalies
for idx, row in df_anomalies.iterrows():
    start_dt = pd.to_datetime(row['anomaly_start'], utc=True)
    end_dt = pd.to_datetime(row['anomaly_end'], utc=True)
    plt.axvspan(start_dt, end_dt, color='red', alpha=0.3, label='Known Anomaly Window' if idx==0 else "")

# Horizontal line for selected threshold (Assuming 0.02 or 2% for the plot, will adjust if another is selected)
selected_threshold = 0.02
plt.axhline(selected_threshold, color='black', linestyle='--', label=f'Selected Threshold ({selected_threshold*100}%)')

plt.title("Anomalous Context Fraction Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Anomalous Context Fraction")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/anomalous_fraction_timeline.png", dpi=150)
plt.close()

# Normal vs Known-Anomaly Distribution
plt.figure(figsize=(8, 6))
sns.boxplot(
    x='known_anomaly', 
    y='anomalous_context_fraction', 
    data=system_df,
    palette=['#a8e6cf', '#ff8b94']
)
plt.title("Distribution of Anomalous Context Fraction (Normal vs Known Anomaly)")
plt.xlabel("Is Known Anomaly Window (0=Normal, 1=Anomaly)")
plt.ylabel("Anomalous Context Fraction")
plt.xticks([0, 1], ['Normal Intervals', 'Known Anomaly Intervals'])
plt.tight_layout()
plt.savefig("outputs/fraction_distribution.png", dpi=150)
plt.close()
print("Plots saved.")
