import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from gan_model import GANModel
from pso_optimizer import PSOOptimizer
from multi_objective_optimizer import MultiObjectiveOptimizer
from integrated_model import IntegratedModel
from dataprocessor import DataPreprocessor



#\subsection{Integration of Models:}
#\begin{itemize}
#\item Integrate the GAN model, PSO for dynamic optimization, and the multi-objective optimization model into a single framework.
#\item Evaluate the performance of the integrated model and compare it with individual components to assess the benefits of the integration.
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

# Train GAN model
gan_model = GANModel(input_shape)
gan_model.train_gan_model(train_data, num_epochs=100, batch_size=32)

# Apply PSO algorithm to optimize GAN parameters
pso_optimizer = PSOOptimizer(gan_model, train_data)
optimized_gan_parameters = pso_optimizer.optimize_gan_parameters()

# Define emission factors and cost coefficients
emission_factors = np.array([0.5, 0.3, 0.2])
cost_coefficients = np.array([0.1, 0.2, 0.3])

# Train multi-objective optimization model
multi_objective_optimizer = MultiObjectiveOptimizer(gan_model, train_data, emission_factors, cost_coefficients)
optimized_multi_objective_parameters = multi_objective_optimizer.optimize_multi_objective()

# Integrate models
integrated_model = IntegratedModel(gan_model, optimized_gan_parameters, optimized_multi_objective_parameters)

# Evaluate integrated model
performance_metrics = integrated_model.evaluate_integrated_model(test_data)
print(f'Integrated model - MAE: {performance_metrics["mae"]}, MSE: {performance_metrics["mse"]}, RMSE: {performance_metrics["rmse"]}, CO2 emissions: {performance_metrics["co2_emissions"]}, energy cost: {performance_metrics["energy_cost"]}')

# Evaluate individual components
y_true = test_data[:, -1]

# Evaluate GAN model
y_pred_gan = gan_model.generator.predict(test_data)
mae_gan = mean_absolute_error(y_true, y_pred_gan)
mse_gan = mean_squared_error(y_true, y_pred_gan)
rmse_gan = np.sqrt(mse_gan)
print(f'GAN model - MAE: {mae_gan}, MSE: {mse_gan}, RMSE: {rmse_gan}')

# Evaluate PSO optimization
optimized_gan_model = GANModel(input_shape)
optimized_gan_model.generator.set_weights(optimized_gan_parameters)
y_pred_pso = optimized_gan_model.generator.predict(test_data)
mae_pso = mean_absolute_error(y_true, y_pred_pso)
mse_pso = mean_squared_error(y_true, y_pred_pso)
rmse_pso = np.sqrt(mse_pso)
print(f'PSO optimization - MAE: {mae_pso}, MSE: {mse_pso}, RMSE: {rmse_pso}')

# Evaluate multi-objective optimization
multi_objective_model = GANModel(input_shape)
multi_objective_model.generator.set_weights(optimized_multi_objective_parameters)
y_pred_multi_objective = multi_objective_model.generator.predict(test_data)
co2_emissions_multi_objective = np.sum(y_pred_multi_objective * emission_factors)
energy_cost_multi_objective = np.sum(y_pred_multi_objective * cost_coefficients)
print(f'Multi-objective optimization model - CO2 emissions: {co2_emissions_multi_objective}, energy cost: {energy_cost_multi_objective}')
