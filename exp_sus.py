import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from gan_model import GANModel
from pso_optimizer import PSOOptimizer
from multi_objective_optimizer import MultiObjectiveOptimizer
from dataprocessor import DataPreprocessor

#\subsection{Sustainability and Efficiency:}
#\begin{itemize}
#\item Investigate the trade-off between CO2 emissions and energy costs by varying the weighting factors in the multi-objective optimization model.
#\item Evaluate the model's ability to reduce CO2 emissions and energy costs while meeting the required energy demand.
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

# Define emission factors and cost coefficients
emission_factors = np.array([0.5, 0.3, 0.2])
cost_coefficients = np.array([0.1, 0.2, 0.3])

# Vary weighting factors in multi-objective optimization model
weighting_factors = np.linspace(0, 1, 11)
co2_emissions = []
energy_costs = []
for w in weighting_factors:
    # Train multi-objective optimization model
    multi_objective_optimizer = MultiObjectiveOptimizer(gan_model, train_data, emission_factors, cost_coefficients, w)
    optimized_multi_objective_parameters = multi_objective_optimizer.optimize_multi_objective()

    # Evaluate multi-objective optimization model
    multi_objective_model = GANModel(input_shape)
    multi_objective_model.generator.set_weights(optimized_multi_objective_parameters)
    y_pred_multi_objective = multi_objective_model.generator.predict(test_data)
    co2_emissions.append(np.sum(y_pred_multi_objective * emission_factors))
    energy_costs.append(np.sum(y_pred_multi_objective * cost_coefficients))

# Plot trade-off between CO2 emissions and energy costs
plt.plot(co2_emissions, energy_costs)
plt.xlabel('CO2 emissions')
plt.ylabel('Energy costs')
plt.title('Trade-off between CO2 emissions and energy costs')
plt.show()

# Evaluate model's ability to reduce CO2 emissions and energy costs
baseline_model = GANModel(input_shape)
baseline_model.generator.set_weights(gan_model.generator.get_weights())
y_pred_baseline = baseline_model.generator.predict(test_data)
co2_emissions_baseline = np.sum(y_pred_baseline * emission_factors)
energy_cost_baseline = np.sum(y_pred_baseline * cost_coefficients)

multi_objective_model = GANModel(input_shape)
multi_objective_model.generator.set_weights(optimized_multi_objective_parameters)
y_pred_multi_objective = multi_objective_model.generator.predict(test_data)
co2_emissions_multi_objective = np.sum(y_pred_multi_objective * emission_factors)
energy_cost_multi_objective = np.sum(y_pred_multi_objective * cost_coefficients)

print(f'Baseline model - CO2 emissions: {co2_emissions_baseline}, energy cost: {energy_cost_baseline}')
print(f'Multi-objective optimization model - CO2 emissions: {co2_emissions_multi_objective}, energy cost: {energy_cost_multi_objective}')
