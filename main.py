import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class EarthquakePredictionModel:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.labels, test_size=0.25
        )

        # Train the model
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def predict(self, new_features):
        """Predicts the probability of an earthquake occurring for the given features."""
        predictions = self.model.predict(new_features)
        return predictions

# Example usage:

# Load the earthquake data
data = pd.read_csv("earthquake_data.csv")

# Extract the features and labels
features = data[["seismic_activity", "ground_deformation", "other_geological_factors"]]
labels = data["magnitude"]

# Create the earthquake prediction model
model = EarthquakePredictionModel(features, labels)

# Make a prediction for a new region
new_features = np.array([[10, 20, 30]])
prediction = model.predict(new_features)

# Print the prediction
print(prediction)