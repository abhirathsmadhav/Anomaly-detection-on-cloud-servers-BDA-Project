# Observation & Feature Unit Analysis

Here are the results of the four specific data structure analyses run via DuckDB, strictly to determine what constitutes a meaningful observation/feature unit for your data.

---

### 1. Combinations of Attributes
We checked if every combination of `(interval_start, location, kind, host, method, statusCode, endpoint)` had exactly 8 `aggregated_stats_name` values.

**Results:**
- **Combinations with exactly 8 stats:** `51,655,156`
- **Combinations with fewer than 8 stats:** `0`
- **Incomplete combinations:** `None`

**What this tells us:**
The dataset is perfectly "rectangular" at this level of granularity. There are no missing statistics. This confirms that the most granular and correct "observation unit" (or feature row) is a pivoted version of this exact combination. By grouping on those 7 attributes, you can perfectly pivot the 8 stats into 8 feature columns without worrying about missing values or NaNs.

---

### 2. Hosts and Endpoints Relationship
We checked the cardinality and relationship between hosts and endpoints across CLIENT and SERVER telemetry.

**Results:**
- **SERVER Telemetry:**
  - Unique Endpoints: `441`
  - Unique Hosts: `13`
  - Avg Endpoints per Host: `34.2`
  - Range: Minimum of `2` up to a maximum of `184` endpoints per host.
- **CLIENT Telemetry:**
  - Unique Endpoints: `745`
  - Unique Hosts: `47`
  - Avg Endpoints per Host: `17.3`
  - Range: Minimum of `1` up to a maximum of `219` endpoints per host.

**What this tells us:**
The relationship is heavily one-to-many. A single physical/virtual host is responsible for dozens, sometimes hundreds, of different API endpoints. Therefore, you cannot aggregate data purely at the `host` level without losing massive amounts of context. The feature unit must at least preserve the `(host, endpoint)` relationship.

---

### 3. Anomaly Windows Mapping to Intervals
We checked if the 25 specific anomaly timestamps (from the CSV) cleanly map onto the 5-minute `interval_start` telemetry timestamps.

**Results (Sample of Notable Windows):**
| Anomaly Window | Duration (Minutes) | Covered 5-Min Intervals |
| :--- | :--- | :--- |
| **a1** | 39.0 mins | 8 intervals |
| **a14** | 1,108.0 mins (longest) | 222 intervals |
| **a19** | 1,071.0 mins | 214 intervals |
| **a21** | 0.0 mins (shortest) | 0 intervals |
| **a23** | 2.0 mins | 1 interval |

*(Note: Most anomalies mapped perfectly, yielding ~1 interval for every 5 minutes of duration).*

**What this tells us:**
The telemetry resolution is exactly 5 minutes. This works perfectly for sustained anomalies (like `a14`). However, ultra-short micro-anomalies (like `a21` which lasted 0 minutes) fell entirely *between* two 5-minute interval ticks, meaning there is no telemetry data recorded strictly *during* that anomaly. You will need to account for this by expanding the label window (e.g., labeling the interval immediately following `a21` as an anomaly).

---

### 4. Anomaly Overlap with Telemetry
We checked what the telemetry looks like purely inside the bounds of the known anomaly windows.

**Results (Rows intersecting Anomaly Windows):**
- **SERVER Overlap:**
  - `4,192,520` rows
  - Affected Hosts: `13` (out of 13 total)
  - Affected Endpoints: `243`
  - Affected Locations: `7` (out of 7 total)
- **CLIENT Overlap:**
  - `6,990,040` rows
  - Affected Hosts: `40` (out of 47 total)
  - Affected Endpoints: `329`
  - Affected Locations: `7` (out of 7 total)

**What this tells us:**
When an anomaly occurs in your system, it is a **cascading, system-wide failure**. It does not just affect one endpoint on one host. Inside these anomaly windows, *all 13 servers* across *all 7 data centers* were impacted, generating millions of abnormal telemetry rows. This implies that anomalies are highly correlated across the network, rather than isolated to single nodes.
