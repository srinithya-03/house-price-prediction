# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("house_price.csv")

# Display first 5 rows
print("Dataset:")
print(data.head())

# Input features
X = data[["Area", "Bedrooms", "Bathrooms", "Age"]]

# Target variable
y = data["Price"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Print actual and predicted prices
print("\nActual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(y_pred)

# Evaluate model
print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict price of a new house
area = float(input("\nEnter Area (sq ft): "))
bedrooms = int(input("Enter Number of Bedrooms: "))
bathrooms = int(input("Enter Number of Bathrooms: "))
age = int(input("Enter House Age (years): "))

new_house = [[area, bedrooms, bathrooms, age]]

predicted_price = model.predict(new_house)

print("\nPredicted House Price: ₹", round(predicted_price[0], 2))