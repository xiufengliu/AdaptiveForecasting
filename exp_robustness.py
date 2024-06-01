import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from gan_model import GANModel
from pso_optimizer import PSOOptimizer
from multi_objective_optimizer import MultiObjectiveOptimizer
from dataprocessor import DataPreprocessor

#\subsection{Robustness and Generalization:}
#\begin{itemize}
#\item Test the model's robustness and generalization by applying it to different datasets and scenarios.
#\item Evaluate the model's performance under different weather conditions and building characteristics.
#\end{itemize}


# Load and preprocess data for different scenarios
data1 = pd.read_csv('energy_data_scenario1.csv')
data2 = pd.read_csv('energy_data_scenario2.csv')
# ...

# Preprocess data for each scenario
preprocessor1 = DataPreprocessor(data1)
data1 = preprocessor1.preprocess_data()
preprocessor2 = DataPreprocessor(data2)
data2 = preprocessor2.preprocess_data()
# ...

# Define input shape for GAN model
input_shape = (data1.shape[1],)

# Train GAN model for each scenario
gan_model1 = GANModel(input_shape)
gan_model1.train_gan_model(data1, num_epochs=100, batch_size=32)
gan_model2 = GANModel(input_shape)
gan_model2.train_gan_model(data2, num_epochs=100, batch_size=32)
# ...

# Define emission factors and cost coefficients
emission_factors = np.array([0.5, 0.3, 0.2])
cost_coefficients = np.array([0.1, 0.2, 0.3])

# Train multi-objective optimization model for each scenario
multi_objective_optimizer1 = MultiObjectiveOptimizer(gan_model1, data1, emission_factors, cost_coefficients)
optimized_multi_objective_parameters1 = multi_objective_optimizer1.optimize_multi_objective()
multi_objective_optimizer2 = MultiObjectiveOptimizer(gan_model2, data2, emission_factors, cost_coefficients)
optimized_multi_objective_parameters2 = multi_objective_optimizer2.optimize_multi_objective()
# ...

# Evaluate model's performance for each scenario
multi_objective_model1 = GANModel(input_shape)
multi_objective_model1.generator.set_weights(optimized_multi_objective_parameters1)
y_pred1 = multi_objective_model1.generator.predict(data1)
mae1 = mean_absolute_error(data1[:, -1], y_pred1)
mse1 = mean_squared_error(data1[:, -1], y_pred1)
rmse1 = np.sqrt(mse1)
co2_emissions1 = np.sum(y_pred1 * emission_factors)
energy_cost1 = np.sum(y_pred1 * cost_coefficients)

multi_objective_model2 = GANModel(input_shape)
multi_objective_model2.generator.set_weights(optimized_multi_objective_parameters2)
y_pred2 = multi_objective_model2.generator.predict(data2)
mae2 = mean_absolute_error(data2[:, -1], y_pred2)
mse2 = mean_squared_error(data2[:, -1], y_pred2)
rmse2 = np.sqrt(mse2)
co2_emissions2 = np.sum(y_pred2 * emission_factors)
energy_cost2 = np.sum(y_pred2 * cost_coefficients)
# ...

# Print performance metrics for each scenario
print(f'Scenario 1 - MAE: {mae1}, MSE: {mse1}, RMSE: {rmse1}, CO2 emissions: {co2_emissions1}, energy cost: {energy_cost1}')
print(f'Scenario 2 - MAE: {mae2}, MSE: {mse2}, RMSE: {rmse2}, CO2 emissions: {co2_emissions2}, energy cost: {energy_cost2}')
# ...
