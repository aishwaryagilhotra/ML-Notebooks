import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Column names (from auto-mpg.names file)
columns = [
    "mpg", "cylinders", "displacement", "horsepower",
    "weight", "acceleration", "model_year", "origin", "car_name"
]

# Load the .data file (space separated)
df = pd.read_csv(
    "auto-mpg.data",
    sep=r"\s+",
    names=columns,
    na_values="?"
)


df = df.dropna()

# Convert horsepower to numeric
df["horsepower"] = pd.to_numeric(df["horsepower"])

# NEGATIVE SLOPE
# Horsepower vs MPG

X_neg = df[["horsepower"]]
Y_neg = df["mpg"]

model_neg = LinearRegression()
model_neg.fit(X_neg, Y_neg)

print("Negative Slope (Horsepower vs MPG):", model_neg.coef_[0])

Y_pred_neg = model_neg.predict(X_neg)

plt.figure()
plt.scatter(X_neg, Y_neg)
plt.plot(X_neg, Y_pred_neg)
plt.title("Horsepower vs MPG (Negative Slope)")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.show()


# POSITIVE SLOPE
# Displacement vs Horsepower

X_pos = df[["displacement"]]
Y_pos = df["horsepower"]

model_pos = LinearRegression()
model_pos.fit(X_pos, Y_pos)

print("Positive Slope (Displacement vs Horsepower):", model_pos.coef_[0])

Y_pred_pos = model_pos.predict(X_pos)

plt.figure()
plt.scatter(X_pos, Y_pos)
plt.plot(X_pos, Y_pred_pos)
plt.title("Displacement vs Horsepower (Positive Slope)")
plt.xlabel("Displacement")
plt.ylabel("Horsepower")
plt.show()
