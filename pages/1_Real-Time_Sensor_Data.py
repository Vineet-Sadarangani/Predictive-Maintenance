import streamlit as st
import pandas as pd
import time

def show_sensor_data(engine_data):
    st.subheader("📈 Real-Time Sensor Dataaaaaaaaaaaa")

    # Identify sensor columns
    sensor_cols = [col for col in engine_data.columns if "s" in col and col.startswith("s")]
    if not sensor_cols:
        st.warning("No sensor columns found in the uploaded dataset.")
        return

    # Dropdown to select a sensor
    selected_sensor = st.selectbox("🔍 Select a Sensor", sensor_cols)

    # Real-Time Graph Simulation
    sensor_data = engine_data[["cycle", selected_sensor]]
    if st.button("▶️ Start Simulation"):
        st.info("Simulation started. Visualizing real-time data...")
        placeholder = st.empty()

        for i in range(1, len(sensor_data)):
            # Display data incrementally
            data_to_display = sensor_data.iloc[:i]
            placeholder.line_chart(data_to_display.set_index("cycle"))

            time.sleep(0.1)  # Add a delay to simulate real-time plotting

def main():
    st.title("📊 Real-Time Sensor Data Visualization")

    # File uploader for engine data
    uploaded_csv = st.file_uploader("Upload Engine Data (CSV)", type=["csv"])
    if uploaded_csv:
        # Load and display the uploaded CSV
        engine_data = pd.read_csv(uploaded_csv)
        st.write("### Uploaded Engine Data")
        st.dataframe(engine_data.head())

        # Show the real-time sensor data visualization
        show_sensor_data(engine_data)
    else:
        st.warning("Please upload a CSV file to visualize sensor data.")

if __name__ == "__main__":
    main()
