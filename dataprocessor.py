import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats

class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def handle_missing_values(self):
        # Handle missing values using mean/median imputation for numerical features and mode imputation for categorical features
        num_imputer = SimpleImputer(strategy='mean')
        cat_imputer = SimpleImputer(strategy='most_frequent')

        num_features = self.data.select_dtypes(include=['int64', 'float64']).columns
        cat_features = self.data.select_dtypes(include=['object']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_imputer, num_features),
                ('cat', cat_imputer, cat_features)
            ])

        self.data = pd.DataFrame(preprocessor.fit_transform(self.data), columns=self.data.columns)

    def normalize_data(self):
        # Normalize data using Min-Max normalization
        scaler = MinMaxScaler()
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)

    def convert_categorical_variables(self):
        # Convert categorical variables to numerical format using one-hot encoding
        cat_features = self.data.select_dtypes(include=['object']).columns
        encoder = OneHotEncoder(drop='first')
        self.data = pd.get_dummies(self.data, columns=cat_features)

    def feature_engineering(self):
        # Create interaction terms
        self.data['interaction'] = self.data['feature1'] * self.data['feature2']

        # Create polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(self.data[['feature1', 'feature2']])
        poly_features = pd.DataFrame(poly_features, columns=['feature1_sq', 'feature1_feature2', 'feature2_sq'])
        self.data = pd.concat([self.data, poly_features], axis=1)

        # Create temporal features
        self.data['hour'] = self.data['timestamp'].dt.hour
        self.data['dayofweek'] = self.data['timestamp'].dt.dayofweek
        self.data['month'] = self.data['timestamp'].dt.month
        self.data['year'] = self.data['timestamp'].dt.year


    def handle_outliers(self):
        # Calculate the Z-scores for each feature
        z_scores = np.abs(stats.zscore(self.data))

        # Remove rows with Z-scores greater than 3 (i.e., outliers)
        self.data = self.data[(z_scores < 3).all(axis=1)]


    def preprocess_data(self):
        # Call all the data preprocessing methods
        self.handle_missing_values()
        self.normalize_data()
        self.convert_categorical_variables()
        self.feature_engineering()
        self.handle_outliers()

        return self.data
