# Structural Statistics of the Parquet Dataset

While PySpark is downloading in the background (it's a 450MB package!), I used a highly optimized analytical database engine called **DuckDB** to query your `413 million` row Parquet file. DuckDB is phenomenal for running SQL queries directly on large Parquet files out-of-core (meaning it doesn't run out of memory).

Here are the structural statistics and distributions of your dataset:

## 1. Overall Size
- **Total Rows:** `413,241,248`

## 2. Row Distribution by `kind`
The dataset is split between client-side metrics and server-side metrics.
| Kind | Row Count | Percentage |
| :--- | :--- | :--- |
| **CLIENT** | `259,441,848` | ~62.7% |
| **SERVER** | `153,799,400` | ~37.3% |

## 3. Row Distribution by `location` (Data Centers)
The metrics are collected across 7 different data centers.
| Location | Row Count |
| :--- | :--- |
| **datacenter4** | `98,005,368` (Largest) |
| **datacenter2** | `89,761,064` |
| **datacenter3** | `86,739,328` |
| **datacenter1** | `72,663,208` |
| **datacenter6** | `48,236,872` |
| **datacenter7** | `10,580,200` |
| **datacenter5** | `7,255,208` (Smallest) |

## 4. Distribution by `aggregated_stats_name`
The dataset calculates 8 different statistical metrics for every single time interval. You have exactly `51,655,156` rows for *each* of the following statistics:
1. `min`
2. `max`
3. `avg`
4. `median`
5. `count`
6. `std` (Standard Deviation)
7. `skewness` (Asymmetry of the data distribution)
8. `kurtosis` (Tails/Outliers of the data distribution)

## 5. Statistical Summary of Values
Looking at the actual numeric values inside the `aggregated_stats_value` column:
- **Global Minimum Value:** `-166.98`
- **Global Average Value:** `566,630.94`

*(Note: The variance here is very high because it mixes counts (e.g., number of requests) with durations (e.g., latency in milliseconds). During your feature engineering phase, you will likely need to normalize or standardize these values!)*

---
> [!TIP]
> **Next Steps for your BDA Project:**
> These structural statistics confirm that your data is highly structured and well-balanced across the different statistical aggregations. Your next step should be **Feature Engineering**: transforming these 8 rows per interval (min, max, avg, etc.) into a single row with 8 columns (pivoting the data) so you can feed it into a Machine Learning model like an Isolation Forest!
