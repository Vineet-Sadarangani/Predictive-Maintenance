import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import timedelta

# Load Dataset
@st.cache_data
def load_data():
    # Replace this with the path to your dataset
    df = pd.read_csv("Dataset\PredictiveManteinanceEngineTraining.csv")
    return df

# Countdown Timer
def format_time(delta):
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days} Days {hours:02} Hours {minutes:02} Minutes {seconds:02} Seconds"

# Main Streamlit App
def main():
    st.title("Engine Health Monitoring Dashboard")

    # Load Dataset
    df = load_data()
    engine_ids = df["id"].unique()

    # Sidebar for engine selection
    st.sidebar.header("Engine Selection")
    engine = st.sidebar.selectbox("Select an Engine", engine_ids, index=0)

    # Filter dataset for the selected engine
    engine_data = df[df["id"] == engine]

    # Remaining Useful Life (RUL)
    rul = engine_data["RUL"].iloc[-1]
    time_remaining = timedelta(days=int(rul))  # Ensure RUL is converted to int

    # Warning Metrics
    warning = timedelta(days=15)
    crash_time = time_remaining - timedelta(days=200)

    # Display Countdown Timers
    st.subheader(f"Engine {engine} Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Remaining Life (RUL)", format_time(time_remaining))
    with col2:
        st.metric("Warning", format_time(warning))
    with col3:
        st.metric("Crash Time", format_time(crash_time))

    # Dynamic Sensor Graphs
    st.subheader("Real-Time Sensor Data")
    sensor_cols = [col for col in engine_data.columns if "s" in col and col.startswith("s")]
    if sensor_cols:  # Check if there are any sensor columns
        selected_sensor = st.selectbox("Select a Sensor", sensor_cols)

        # Display Real-Time Graph
        sensor_data = engine_data[["cycle", selected_sensor]]
        st.line_chart(sensor_data.set_index("cycle"))

        # Simulate Real-Time Updates
        st.subheader("Simulating Real-Time Updates")
        placeholder = st.empty()
        for i in range(1, len(sensor_data)):
            data_to_display = sensor_data.iloc[:i]
            with placeholder:
                st.line_chart(data_to_display.set_index("cycle"))
            time.sleep(0.1)

    # Timer for Remaining Time
    st.subheader("Countdown Timer")
    countdown_placeholder = st.empty()
    start_time = time.time()
    while rul > 0:
        elapsed = time.time() - start_time
        remaining = time_remaining - timedelta(seconds=elapsed)
        if remaining.total_seconds() > 0:
            countdown_placeholder.metric("Remaining Time", format_time(remaining))
        else:
            countdown_placeholder.metric("Remaining Time", "Expired")
            break
        time.sleep(1)

if __name__ == "__main__":
    main()
