# Predictive-Maintenance
In this example, an LSTM network is developed to predict the remaining useful life (RUL) or time to failure of aircraft engines. The network leverages simulated aircraft sensor data to forecast future engine failures, enabling proactive maintenance planning. The key question is: "Given the operational history and failure events of aircraft engines, can we predict when an in-service engine will fail?" To address this, a Binary Classification model is employed.

The Dataset directory contains training, testing, and ground truth datasets. The training data comprises multiple multivariate time series with "cycle" as the time unit and 21 sensor readings per cycle, each representing a different engine of the same type. The testing data follows the same structure but does not specify the failure point. Meanwhile, the ground truth data provides the remaining operational cycles for the engines in the testing dataset.
