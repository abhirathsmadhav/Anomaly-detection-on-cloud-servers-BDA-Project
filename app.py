import streamlit as st
import pandas as pd
import json
import time
import os

st.set_page_config(
    page_title="Cloud Anomaly Detection",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: 500; }
    .alert-box {
        padding: 15px; background-color: #ff4b4b; color: white;
        border-radius: 5px; margin-bottom: 15px; font-weight: bold;
        text-align: center; animation: blinker 1s linear infinite;
    }
    @keyframes blinker { 50% { opacity: 0.5; } }
</style>
""", unsafe_allow_html=True)

st.title("☁️ Cloud Server Anomaly Detection")
st.markdown("<p class='big-font'>Big Data Analytics Project Demonstration</p>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("📂 Dataset Selection")
    dataset_option = st.sidebar.radio(
        "Choose Data Source to Simulate:",
        ("Targeted Demo (Event a24, ±6 Hours)", "Full Real Dataset (3 Months Testing)")
    )
    
    st.divider()
    
    st.header("⚙️ System Metrics")
    st.markdown("**Model:** High-Dimensional Deep Autoencoder (128>64>14>64>128)")
    st.markdown("**Features:** 5XX HTTP Status Telemetry")
    st.markdown("**Data Processed:** 413 Million Rows")
    st.divider()
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Precision", "4.11%")
    col2.metric("Recall", "18.94%")
    st.metric("F1 Score", "6.75%")
    st.metric("Detected Outages", "9 / 19")

@st.cache_data
def load_data(option):
    if option == "Targeted Demo (Event a24, ±6 Hours)":
        csv_path = 'outputs/demo_selected_anomaly.csv'
        if not os.path.exists(csv_path): return None
        df = pd.read_csv(csv_path)
        df['interval_time'] = pd.to_datetime(df['interval_time'])
        return df
    else:
        pq_path = 'outputs/final_ann_interval_results.parquet'
        if not os.path.exists(pq_path): return None
        df = pd.read_parquet(pq_path)
        df['interval_time'] = pd.to_datetime(df['timestamp'])
        df['reconstruction_error'] = df['mse']
        df['predicted_anomaly'] = df['pred_anomaly']
        df['anomaly_threshold'] = 0.007075717225952203
        return df[['interval_time', 'reconstruction_error', 'anomaly_threshold', 'predicted_anomaly']]

df = load_data(dataset_option)
if df is None:
    st.error("Error: Output files not found. Please ensure scripts have been run.")
    st.stop()

if 'dataset' not in st.session_state:
    st.session_state.dataset = dataset_option
    st.session_state.step = len(df)
    st.session_state.playing = False

if st.session_state.dataset != dataset_option:
    st.session_state.dataset = dataset_option
    st.session_state.step = len(df)
    st.session_state.playing = False

# --- TABS FOR ORGANIZED PRESENTATION ---
tab_sim, tab_eval, tab_data = st.tabs([
    "🔴 Live Simulation", 
    "📊 Model Evaluation & Plots", 
    "📁 Project Outputs"
])

# ==========================================
# TAB 1: LIVE SIMULATION
# ==========================================
with tab_sim:
    st.subheader(f"📡 Real-Time Telemetry Simulation: {dataset_option}")
    st.write("Scrub through the timeline to see the Deep Autoencoder detect anomalies in real-time.")

    col_play, col_speed = st.columns([1, 3])
    if col_play.button("▶️ Auto-Play Simulation"):
        st.session_state.step = 2
        st.session_state.playing = True
        st.rerun()

    if col_play.button("⏹️ Stop Simulation"):
        st.session_state.playing = False
        st.session_state.step = len(df)
        st.rerun()

    max_speed = 5 if len(df) < 1000 else 200
    speed = col_speed.slider("Playback Speed (intervals per frame)", min_value=1, max_value=max_speed, value=max_speed//2)
    manual_step = st.slider("Timeline Scrubber", min_value=1, max_value=len(df), value=st.session_state.step, key="scrubber")

    if manual_step != st.session_state.step and not st.session_state.playing:
        st.session_state.step = manual_step

    current_data = df.iloc[:st.session_state.step]
    latest_row = current_data.iloc[-1]
    threshold = latest_row['anomaly_threshold']

    metrics_placeholder = st.container()
    alert_placeholder = st.empty()
    chart_placeholder = st.empty()

    chart_df = current_data[['interval_time', 'reconstruction_error', 'anomaly_threshold']].set_index('interval_time')
    chart_placeholder.line_chart(chart_df, color=["#1f77b4", "#000000"])

    timestamp_str = latest_row['interval_time'].strftime("%Y-%m-%d %H:%M UTC")
    mse_str = f"{latest_row['reconstruction_error']:.5f}"
    is_anomaly = latest_row['predicted_anomaly'] == 1

    if is_anomaly:
        status_text = "🚨 CRITICAL ANOMALY"
        status_color = "#ff4b4b"
        mse_color = "#ff4b4b"
        bg_style = "background-color: #ffeaea; animation: pulse 1.5s infinite;"
    else:
        status_text = "✅ SYSTEM NORMAL"
        status_color = "#198754"
        mse_color = "#1f77b4"
        bg_style = "background-color: #f8f9fa;"

    metrics_html = f"""
    <div style='display: flex; justify-content: space-between; gap: 20px; margin-bottom: 25px;'>
        <div style='flex: 1; padding: 20px; border-radius: 8px; background-color: #f8f9fa; text-align: center; border-left: 5px solid #6c757d; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
            <p style='margin:0; color:#6c757d; font-size:12px; font-weight:bold; letter-spacing: 1px; text-transform: uppercase;'>Current Timestamp</p>
            <p style='margin:5px 0 0 0; font-size:24px; font-weight:bold; color:#212529;'>{timestamp_str}</p>
        </div>
        <div style='flex: 1; padding: 20px; border-radius: 8px; background-color: #f8f9fa; text-align: center; border-left: 5px solid {mse_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
            <p style='margin:0; color:#6c757d; font-size:12px; font-weight:bold; letter-spacing: 1px; text-transform: uppercase;'>Reconstruction Error (MSE)</p>
            <p style='margin:5px 0 0 0; font-size:24px; font-weight:bold; color:{mse_color};'>{mse_str}</p>
            <p style='margin:0; font-size:11px; color:#adb5bd; margin-top:2px;'>Threshold Limit: {threshold:.5f}</p>
        </div>
        <div style='flex: 1; padding: 20px; border-radius: 8px; text-align: center; border-left: 5px solid {status_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.05); {bg_style}'>
            <style>
                @keyframes pulse {{
                    0% {{ box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.4); }}
                    70% {{ box-shadow: 0 0 0 10px rgba(255, 75, 75, 0); }}
                    100% {{ box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }}
                }}
            </style>
            <p style='margin:0; color:#6c757d; font-size:12px; font-weight:bold; letter-spacing: 1px; text-transform: uppercase;'>Current Status</p>
            <p style='margin:5px 0 0 0; font-size:24px; font-weight:bold; color:{status_color};'>{status_text}</p>
        </div>
    </div>
    """
    with metrics_placeholder.container():
        st.markdown(metrics_html, unsafe_allow_html=True)

    if latest_row['predicted_anomaly'] == 1:
        alert_placeholder.markdown("<div class='alert-box'>🔥 5XX AUTOENCODER RECONSTRUCTION FAILURE DETECTED 🔥</div>", unsafe_allow_html=True)
    else:
        alert_placeholder.empty()

    if st.session_state.playing:
        st.session_state.step += speed
        if st.session_state.step >= len(df):
            st.session_state.step = len(df)
            st.session_state.playing = False
        time.sleep(0.05)
        st.rerun()

# ==========================================
# TAB 2: EVALUATION PLOTS
# ==========================================
with tab_eval:
    st.header("📊 Deep Autoencoder Static Evaluation")
    st.write("These charts visualize the final evaluation over the entire 3-month testing period.")
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists('outputs/final_reconstruction_error.png'):
            st.image('outputs/final_reconstruction_error.png', caption='Reconstruction Error (MSE) over Test Period', use_container_width=True)
        if os.path.exists('outputs/final_confusion_matrix.png'):
            st.image('outputs/final_confusion_matrix.png', caption='Test Period Confusion Matrix', use_container_width=True)
            
    with col2:
        if os.path.exists('outputs/final_5xx_timeline.png'):
            st.image('outputs/final_5xx_timeline.png', caption='Raw 5XX Request Volume Timeline', use_container_width=True)
        if os.path.exists('outputs/final_training_loss.png'):
            st.image('outputs/final_training_loss.png', caption='Autoencoder Training Convergence Loss', use_container_width=True)

# ==========================================
# TAB 3: DATA & OUTPUTS
# ==========================================
with tab_data:
    st.header("📁 Project Outputs & Metrics")
    
    st.subheader("Event-Level Detection Results (Ground Truth mapping)")
    st.write("This table proves which exact documented system outages were successfully detected by the autoencoder.")
    if os.path.exists('outputs/final_ann_event_results.csv'):
        events_df = pd.read_csv('outputs/final_ann_event_results.csv')
        st.dataframe(events_df, use_container_width=True)
        
    st.subheader("Algorithm Thresholding Evaluation")
    st.write("Comparison of Baseline Raw-MSE vs temporal likelihood methods.")
    if os.path.exists('outputs/final_ann_results.json'):
        with open('outputs/final_ann_results.json', 'r') as f:
            metrics_data = json.load(f)
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
    st.subheader("Temporal Refinement Evaluation")
    st.write("Comparison showing that temporal post-processing smoothing drastically increased false positives.")
    if os.path.exists('outputs/final_ann_refined_comparison.csv'):
        refined_df = pd.read_csv('outputs/final_ann_refined_comparison.csv')
        st.dataframe(refined_df, use_container_width=True)
