
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("house_price.csv")

print("Dataset:")
print(data.head())

X = data[["Area", "Bedrooms", "Bathrooms", "Age"]]


y = data["Price"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\nActual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(y_pred)


print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

area = float(input("\nEnter Area (sq ft): "))
bedrooms = int(input("Enter Number of Bedrooms: "))
bathrooms = int(input("Enter Number of Bathrooms: "))
age = int(input("Enter House Age (years): "))

new_house = [[area, bedrooms, bathrooms, age]]

predicted_price = model.predict(new_house)

print("\nPredicted House Price: ₹", round(predicted_price[0], 2))