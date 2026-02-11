import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

columns = [
    "mpg", "cylinders", "displacement", "horsepower",
    "weight", "acceleration", "model_year", "origin", "car_name"
]

df = pd.read_csv(
    "auto-mpg.data",
    sep=r"\s+",
    names=columns,
    na_values="?"
)

df = df.dropna()
df["horsepower"] = pd.to_numeric(df["horsepower"])


X = df["horsepower"].values
Y = df["mpg"].values

# Normalize
X = (X - X.mean()) / X.std()

# Initialize parameters
w = 0
b = 0
learning_rate = 0.01
max_epochs = 5000
tolerance = 1e-6
n = len(X)

loss_history = []


# Gradient Descent

for epoch in range(max_epochs):
    Y_pred = w * X + b

    # Calculate loss
    loss = np.mean((Y - Y_pred) ** 2)
    loss_history.append(loss)

    # Compute gradients
    dw = (-2/n) * np.sum(X * (Y - Y_pred))
    db = (-2/n) * np.sum(Y - Y_pred)

    # Update
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Print every 50 iterations
    if epoch % 50 == 0:
        print(f"Iteration {epoch}, Loss: {loss}")

    # Convergence check
    if epoch > 0 and abs(loss_history[-2] - loss) < tolerance:
        print("Converged at iteration:", epoch)
        break

# Plot Loss Curve

plt.figure()
plt.plot(loss_history)
plt.title("Loss vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("MSE Loss")
plt.show()
