import pandas as pd
import matplotlib.pyplot as plt

# Function to plot sensor data for a specific ID
def plot_sensor_data(dataframe, machine_id, sensor):
    """
    Plots sensor data over time (cycle) for a specific machine ID.
    
    Parameters:
        dataframe (pd.DataFrame): The dataset containing the sensor data.
        machine_id (int): The ID of the machine to filter.
        sensor (str): The sensor column name (e.g., 's1').
    """
    if sensor not in dataframe.columns:
        raise ValueError(f"Sensor '{sensor}' not found in the dataframe.")
    
    # Filter data for the given ID
    machine_data = dataframe[dataframe['id'] == machine_id]
    
    # Plot the sensor data over cycle
    plt.figure(figsize=(10, 6))
    plt.plot(machine_data['cycle'], machine_data[sensor], marker='o', linestyle='-', color='blue', label=f'Sensor {sensor}')
    plt.title(f'Sensor {sensor} over Time (Cycle) for ID={machine_id}')
    plt.xlabel('Cycle')
    plt.ylabel(f'Sensor {sensor} Value')
    plt.grid(True)
    plt.legend()
    plt.show()

test_df = pd.read_csv(r'Dataset\PM_test.txt', sep=" ", header=None)
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']

plot_sensor_data(test_df, machine_id=2, sensor='s3')