import streamlit as st
from datetime import datetime, timedelta
import time

# Format countdown timers beautifully
def format_time(delta):
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"""
    <div style='font-size: 30px; font-weight: bold; color: #ff4b4b; text-align: center; background-color: #333;
                padding: 10px; border-radius: 10px; box-shadow: 0 0 10px rgba(255, 75, 75, 0.5);'>
        {days} <span style='color: #FFC1C3;'>Days</span>
        {hours:02} <span style='color: #FFC1C3;'>Hours</span>
        {minutes:02} <span style='color: #FFC1C3;'>Minutes</span>
        {seconds:02} <span style='color: #FFC1C3;'>Seconds</span>
    </div>
    """

def main():
    st.title("🚨 Engine Health Monitoring with Alerts")

    # Dummy values for testing (RUL set to trigger within minutes or hours)
    selected_engine = 1
    predicted_rul = 0.03  # Predicted RUL in days (~43 minutes)
    initial_rul = timedelta(days=predicted_rul)
    warning_time = timedelta(days=0.6 * predicted_rul)  # 60% of RUL
    crash_time = timedelta(days=0.8 * predicted_rul)  # 80% of RUL

    # Timers for testing
    if "engine_timers" not in st.session_state:
        st.session_state.engine_timers = {}

    if selected_engine not in st.session_state.engine_timers:
        st.session_state.engine_timers[selected_engine] = {
            "rul_end_time": datetime.now() + initial_rul,
            "warning_end_time": datetime.now() + warning_time,
            "crash_end_time": datetime.now() + crash_time
        }

    timers = st.session_state.engine_timers[selected_engine]
    rul_placeholder = st.empty()
    warning_placeholder = st.empty()
    crash_placeholder = st.empty()
    alert_placeholder = st.empty()  # Placeholder for the pop-up message

    st.write(f"Monitoring Engine: {selected_engine}")

    # Countdown timer loop
    for _ in range(10000):
        now = datetime.now()

        remaining_rul = max(timers["rul_end_time"] - now, timedelta(0))
        remaining_warning = max(timers["warning_end_time"] - now, timedelta(0))
        remaining_crash = max(timers["crash_end_time"] - now, timedelta(0))

        rul_placeholder.markdown(f"Predicted RUL Timer:<br>{format_time(remaining_rul)}", unsafe_allow_html=True)
        warning_placeholder.markdown(f"Warning Timer:<br>{format_time(remaining_warning)}", unsafe_allow_html=True)
        crash_placeholder.markdown(f"Crash Timer:<br>{format_time(remaining_crash)}", unsafe_allow_html=True)

        # Simulated pop-up message
        if remaining_rul <= timedelta(hours=1) and remaining_rul > timedelta(0):
            with alert_placeholder.container():
                st.markdown(
                    """
                    <div style='font-size: 20px; font-weight: bold; color: #FFFFFF; text-align: center;
                                background-color: #FF0000; padding: 20px; border-radius: 10px;'>
                        ⚠ ALERT: Remaining RUL is less than 1 hour! Take immediate action.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            alert_placeholder.empty()  # Clear the alert when not needed

        if remaining_rul == timedelta(0):
            st.write("⏹ Timer has ended!")
            break

        time.sleep(1)


if __name__ == "__main__":
    main()
