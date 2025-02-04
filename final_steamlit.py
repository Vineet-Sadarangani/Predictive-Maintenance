import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from keras.models import load_model
import tensorflow.keras.backend as K
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Custom R² metric
def r2_keras(y_true, y_pred):
    """Coefficient of Determination"""
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))

# Load pre-trained model
model_path = "regression_model.keras"
estimator = load_model(model_path, custom_objects={'r2_keras': r2_keras})

# Function to preprocess the test data
def preprocess_data(test_file, truth_file):
    test_df = pd.read_csv(test_file, sep=" ", header=None)
    test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
    test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                       's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                       's15', 's16', 's17', 's18', 's19', 's20', 's21']

    truth_df = pd.read_csv(truth_file, sep=" ", header=None)
    truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
    truth_df.columns = ['more']
    truth_df['id'] = truth_df.index + 1

    rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    truth_df['max'] = rul['max'] + truth_df['more']
    test_df = test_df.merge(truth_df[['id', 'max']], on='id', how='left')
    test_df['RUL'] = test_df['max'] - test_df['cycle']

    cols_normalize = test_df.columns.difference(['id', 'cycle', 'RUL'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_test_df = pd.DataFrame(min_max_scaler.fit_transform(test_df[cols_normalize]),
                                columns=cols_normalize,
                                index=test_df.index)
    test_df = test_df[['id', 'cycle', 'RUL']].join(norm_test_df)
    test_df['cycle_norm'] = test_df['cycle'] / test_df.groupby('id')['cycle'].transform('max')

    return test_df

# Prepare data for prediction
def prepare_sequences(test_df, sequence_length=31):
    sensor_cols = ['s' + str(i) for i in range(1, 22)]
    sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm'] + sensor_cols

    seq_array_test_last = [test_df[test_df['id'] == id][sequence_cols].values[-sequence_length:]
                           for id in test_df['id'].unique() if len(test_df[test_df['id'] == id]) >= sequence_length]
    seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

    y_mask = [len(test_df[test_df['id'] == id]) >= sequence_length for id in test_df['id'].unique()]
    label_array_test_last = test_df.groupby('id')['RUL'].nth(-1)[y_mask].values
    label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(np.float32)

    return seq_array_test_last, label_array_test_last

# Format time for countdown timers
def format_time(delta):
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"""
    <div style='font-size: 24px; font-weight: bold; color: #ffffff; text-align: center; 
                background-color: #222; padding: 15px; border-radius: 8px; 
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);'>
        {days} <span style='color: #FFD700;'>Days</span>, 
        {hours:02} <span style='color: #FFD700;'>Hours</span>, 
        {minutes:02} <span style='color: #FFD700;'>Minutes</span>, 
        {seconds:02} <span style='color: #FFD700;'>Seconds</span>
    </div>
    """

# Create a simple line chart of predicted RUL over cycles
def plot_rul_chart(test_df, predicted_rul):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(test_df['cycle'], test_df['RUL'], label='Actual RUL', color='blue')
    ax.axvline(x=test_df[test_df['id'] == selected_engine]['cycle'].max(), color='red', linestyle='--', label='Predicted RUL')
    ax.set_title(f"Predicted RUL for Engine {selected_engine}", fontsize=16)
    ax.set_xlabel("Cycle", fontsize=12)
    ax.set_ylabel("RUL (Days)", fontsize=12)
    ax.legend(loc='upper right')
    st.pyplot(fig)

# Main app
def main():
    st.title("🚀 Engine Health Monitoring Dashboard")

    # Sidebar for file uploads
    uploaded_test_file = st.sidebar.file_uploader("📂 Upload Test Dataset", type="txt", key="test")
    uploaded_truth_file = st.sidebar.file_uploader("📂 Upload Truth Dataset", type="txt", key="truth")

    if uploaded_test_file and uploaded_truth_file:
        # Preprocess data and prepare sequences
        test_df = preprocess_data(uploaded_test_file, uploaded_truth_file)
        seq_array_test_last, label_array_test_last = prepare_sequences(test_df)

        # Sidebar for engine selection
        st.sidebar.header("Engine Selection")
        engine_ids = test_df["id"].unique()
        selected_engine = st.sidebar.selectbox("Select an Engine", engine_ids, index=0)

        st.subheader(f"Engine {selected_engine} Predictions")

        # Prediction for all engines
        y_pred_test = estimator.predict(seq_array_test_last)

        # Get prediction for the selected engine
        engine_index = np.where(engine_ids == selected_engine)[0][0]  # Index of selected engine
        predicted_rul = y_pred_test[engine_index][0]

        # Calculate timers
        initial_rul = timedelta(days=int(predicted_rul))
        warning_time = timedelta(days=0.6 * initial_rul.days)
        crash_time = timedelta(days=0.8 * initial_rul.days)

        # Update session state for selected engine
        if "engine_timers" not in st.session_state:
            st.session_state.engine_timers = {}

        if selected_engine not in st.session_state.engine_timers:
            st.session_state.engine_timers[selected_engine] = {
                "rul_end_time": datetime.now() + initial_rul,
                "warning_end_time": datetime.now() + warning_time,
                "crash_end_time": datetime.now() + crash_time
            }

        timers = st.session_state.engine_timers[selected_engine]

        st.markdown("### Dynamic Countdown Timers")
        rul_placeholder = st.empty()
        warning_placeholder = st.empty()
        crash_placeholder = st.empty()

        # Countdown timer
        while True:
            now = datetime.now()

            remaining_rul = max(timers["rul_end_time"] - now, timedelta(0))
            remaining_warning = max(timers["warning_end_time"] - now, timedelta(0))
            remaining_crash = max(timers["crash_end_time"] - now, timedelta(0))

            rul_placeholder.markdown(
                f"**Predicted RUL Timer:**<br>{format_time(remaining_rul)}", 
                unsafe_allow_html=True
            )
            warning_placeholder.markdown(
                f"**Warning Timer:**<br>{format_time(remaining_warning)}", 
                unsafe_allow_html=True
            )
            crash_placeholder.markdown(
                f"**Crash Timer:**<br>{format_time(remaining_crash)}", 
                unsafe_allow_html=True
            )

            if remaining_rul == timedelta(0):
                st.error("⏹️ Timer has ended!")
                break

            time.sleep(1)

        # Display the predicted RUL chart
        plot_rul_chart(test_df, predicted_rul)

    else:
        st.warning("Please upload both test and truth datasets to proceed.")


if __name__ == "__main__":
    main()
