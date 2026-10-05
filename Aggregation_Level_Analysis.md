# Aggregation Level & Feature Analysis

Here are the results of aggregating the 413-million row dataset into system-wide 5-minute intervals to compare normal system behavior versus the 25 known anomaly windows.

---

### 1. Temporal Continuity
The dataset spans exactly 4 months (Jan 22 to Jun 7).
- **Expected 5-minute intervals:** `39,380`
- **Actual intervals present:** `39,365`
- **Missing intervals:** `15`

*Finding:* The dataset is incredibly continuous. Missing only 15 ticks over 4 months means you do not need to worry about heavy imputation for missing timestamps. 

---

### 2. Normal vs. Anomaly System Behavior
We grouped the data into 38,407 "Normal" intervals and 958 "Anomaly" intervals. Here is how the system-level features behaved:

| Feature | Mean (Normal) | Mean (Anomaly) | Difference |
| :--- | :--- | :--- | :--- |
| **Total Request Volume** | `80,987` | `98,381` | ⬆️ **+21.4%** |
| **Client Volume** | `40,080` | `49,118` | ⬆️ **+22.5%** |
| **Server Volume** | `40,907` | `49,263` | ⬆️ **+20.4%** |
| **Active Hosts** | `37.7` | `37.2` | ~ No Change |
| **Active Endpoints** | `251.0` | `260.7` | ⬆️ Slight Increase |
| **Average Latency** | `856,999` | `942,153` | ⬆️ **+9.9%** |

#### A. Informative Features
- **Volume Metrics (`total_count`, `client_vol`, `server_vol`):** These are highly informative. System volume reliably spikes by over 20% during anomaly windows.
- **Latency (`mean_latency`):** Average latency increases by ~10% during anomalies, indicating degraded system performance.

#### B. Features to Avoid (Ambiguous/Noisy)
- **`max_latency` and `max_total_vol`:** Do not use maximums aggregated at the system level. The max latency during a *normal* period hit 9,090,736,574 (an insane outlier), while the max during an anomaly was only 3,600,036,361. Aggregating `max()` across the entire system makes the feature hyper-sensitive to single, random outliers.

---

### 3. Final Recommendation

**Question:** Should the final anomaly detector operate at:
A) individual host/endpoint level,
B) system-wide 5-minute interval level,
C) a hierarchical approach combining both.

**Recommendation:** **Option C (A hierarchical approach combining both).**

**Why?**
Our analysis proves that Option B (System-wide aggregation) works: the global mean volume and latency *do* visibly increase during anomalies. 

However, collapsing 54 hosts and 1,000+ endpoints into a single global number severely dilutes the signal. If global volume spikes by 20%, it is highly likely that this is caused by a massive 500% spike on just one or two specific failing hosts, while the other 50 hosts remained perfectly normal. 

If you build your model strictly at the system level (Option B), you will detect the anomaly, but you won't know *which* server or endpoint caused it. 

**The Ideal Architecture:**
1. **Primary Observation Unit:** Pivot the data at the `(interval_start, host, endpoint)` level.
2. **Context Features:** Calculate the system-wide 5-minute aggregates (like Total System Volume) and append them as extra columns to the primary observation unit. 
3. **Result:** The model can look at a specific endpoint and ask: *"Is this endpoint spiking because it's failing, or is it spiking because the entire global system is under heavy load?"*
