import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from gan_model import GANModel
from pso_optimizer import PSOOptimizer
from multi_objective_optimizer import MultiObjectiveOptimizer
from dataprocessor import DataPreprocessor

#\subsection{Dynamic Optimization and Multi-objective Optimization:}
#\begin{itemize}
#\item Apply the PSO algorithm to dynamically optimize the GAN parameters to maintain high accuracy over time while minimizing CO2 emissions and energy costs.
#\item Evaluate the performance of the optimized GAN model and compare it with the original GAN model.
#\item Compare the performance of the multi-objective optimization model with a single-objective optimization model that focuses only on minimizing energy costs.
#\end{itemize}



# Load and preprocess data
data = pd.read_csv('energy_data.csv')
preprocessor = DataPreprocessor(data)
data = preprocessor.preprocess_data()

# Split data into training and testing sets
train_data = data[:int(len(data)*0.8)]
test_data = data[int(len(data)*0.8):]

# Scale data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Define input shape for GAN model
input_shape = (train_data.shape[1],)

# Train original GAN model
gan_model = GANModel(input_shape)
gan_model.train_gan_model(train_data, num_epochs=100, batch_size=32)

# Evaluate original GAN model
y_pred_original = gan_model.generator.predict(test_data)
y_true = test_data[:, -1]
mae_original = mean_absolute_error(y_true, y_pred_original)
mse_original = mean_squared_error(y_true, y_pred_original)
rmse_original = np.sqrt(mse_original)
print(f'Original GAN model - MAE: {mae_original}, MSE: {mse_original}, RMSE: {rmse_original}')

# Apply PSO algorithm to optimize GAN parameters
pso_optimizer = PSOOptimizer(gan_model, train_data)
optimized_gan_parameters = pso_optimizer.optimize_gan_parameters()

# Train optimized GAN model
optimized_gan_model = GANModel(input_shape)
optimized_gan_model.generator.set_weights(optimized_gan_parameters)
optimized_gan_model.train_gan_model(train_data, num_epochs=100, batch_size=32)

# Evaluate optimized GAN model
y_pred_optimized = optimized_gan_model.generator.predict(test_data)
mae_optimized = mean_absolute_error(y_true, y_pred_optimized)
mse_optimized = mean_squared_error(y_true, y_pred_optimized)
rmse_optimized = np.sqrt(mse_optimized)
print(f'Optimized GAN model - MAE: {mae_optimized}, MSE: {mse_optimized}, RMSE: {rmse_optimized}')

# Define emission factors and cost coefficients
emission_factors = np.array([0.5, 0.3, 0.2])
cost_coefficients = np.array([0.1, 0.2, 0.3])

# Train multi-objective optimization model
multi_objective_optimizer = MultiObjectiveOptimizer(gan_model, train_data, emission_factors, cost_coefficients)
optimized_multi_objective_parameters = multi_objective_optimizer.optimize_multi_objective()

# Train single-objective optimization model
single_objective_optimizer = PSOOptimizer(gan_model, train_data, cost_coefficients)
optimized_single_objective_parameters = single_objective_optimizer.optimize_gan_parameters()

# Evaluate multi-objective optimization model
multi_objective_model = GANModel(input_shape)
multi_objective_model.generator.set_weights(optimized_multi_objective_parameters)
y_pred_multi_objective = multi_objective_model.generator.predict(test_data)
co2_emissions_multi_objective = np.sum(y_pred_multi_objective * emission_factors)
energy_cost_multi_objective = np.sum(y_pred_multi_objective * cost_coefficients)
print(f'Multi-objective optimization model - CO2 emissions: {co2_emissions_multi_objective}, energy cost: {energy_cost_multi_objective}')

# Evaluate single-objective optimization model
single_objective_model = GANModel(input_shape)
single_objective_model.generator.set_weights(optimized_single_objective_parameters)
y_pred_single_objective = single_objective_model.generator.predict(test_data)
energy_cost_single_objective = np.sum(y_pred_single_objective * cost_coefficients)
print(f'Single-objective optimization model - energy cost: {energy_cost_single_objective}')
