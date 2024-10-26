import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class RegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.r_squared = None
        self.adjusted_r_squared = None

    def load_data(self):
        # Load the California housing dataset from a CSV file
        url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
        data = pd.read_csv(url)

        # Prepare features and target
        X = data.drop("median_house_value", axis=1)  # Use all columns except the target
        y = data["median_house_value"]  # Target variable

        # One-hot encode categorical features
        X = pd.get_dummies(X, drop_first=True)

        # Split the data into train, validation, and test sets
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Validation set: {self.X_val.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")

    def train_model(self):
        # Train the linear regression model
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Make predictions on the validation set
        y_val_pred = self.model.predict(self.X_val)

        # Calculate R-squared and Adjusted R-squared
        self.r_squared = r2_score(self.y_val, y_val_pred)
        n = self.X_val.shape[0]  # number of observations
        p = self.X_val.shape[1]  # number of predictors
        self.adjusted_r_squared = 1 - (1 - self.r_squared) * (n - 1) / (n - p - 1)

        print(f"R-squared: {self.r_squared:.4f}")
        print(f"Adjusted R-squared: {self.adjusted_r_squared:.4f}")

    def plot_results(self):
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_val, self.model.predict(self.X_val), alpha=0.7)
        plt.plot([self.y_val.min(), self.y_val.max()], [self.y_val.min(), self.y_val.max()], color='red', lw=2)
        plt.title("Actual vs Predicted Values")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.grid()
        plt.show()

    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()
        self.plot_results()


# Running the Regression Model
if __name__ == "__main__":
    regression_model = RegressionModel()
    regression_model.run()
