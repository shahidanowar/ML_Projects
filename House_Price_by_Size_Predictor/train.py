import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Load data (uses only 'area' and 'price')
df = pd.read_csv("Housing.csv")
df = df[['area', 'price']].dropna().copy()

X_raw = df[['area']].values.astype(float)  # shape (n_samples, 1)
y = df['price'].values.astype(float)       # shape (n_samples,)

# 2) Standardize the feature (helps gradient descent)
x_mean = X_raw.mean()
x_std = X_raw.std() if X_raw.std() != 0 else 1.0
X = (X_raw - x_mean) / x_std

# 3) Batch Gradient Descent for Linear Regression
def batch_gradient_descent(X, y_true, epochs=10, learning_rate=0.01, verbose=True):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # weights
    b = 0.0                   # bias

    mse_list = []
    r2_list = []
    epoch_list = []

    y_mean = y_true.mean()
    ss_tot = np.sum((y_true - y_mean) ** 2)  # constant across epochs

    for i in range(1, epochs + 1):
        y_pred = X.dot(w) + b
        error = y_pred - y_true

        # gradients (MSE)
        grad_w = (2.0 / n_samples) * (X.T.dot(error))
        grad_b = (2.0 / n_samples) * np.sum(error)

        # update
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        # metrics
        mse = np.mean(error ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        mse_list.append(mse)
        r2_list.append(r2)
        epoch_list.append(i)

        if verbose:
            print(f"Epoch {i:02d} | MSE: {mse:.2f} | R^2: {r2:.4f}")

    return w, b, mse_list, r2_list, epoch_list

# 4) Train (10 epochs per the assignment)
EPOCHS = 20         # try 500–1000
LEARNING_RATE = 0.05  # try 0.05; if unstable, reduce to 0.02 or 0.01
w, b, mse_list, r2_list, epoch_list = batch_gradient_descent(
    X, y, epochs=EPOCHS, learning_rate=LEARNING_RATE, verbose=True
)

# Convert parameters back to original 'area' units (optional but nice)
# If x_std = (x - mean)/std, then:
# y = (w/std)*x + (b - w*mean/std)
w_orig = w[0] / x_std
b_orig = b - (w[0] * x_mean / x_std)

print("\nLearned Parameters (standardized feature):")
print(f"  w (slope on standardized area): {w[0]:.6f}")
print(f"  b (intercept on standardized area): {b:.6f}")

print("\nLearned Parameters (original feature scale):")
print(f"  Slope (price per square ft): {w_orig:.6f}")
print(f"  Intercept: {b_orig:.2f}")

# 5) Predictions for plotting the regression line
y_pred_train = (X.dot(w) + b)  # predictions correspond to X_raw order

# 6) Plots

# 6a) Epoch vs Accuracy (R^2)
plt.figure(figsize=(6,4))
plt.plot(epoch_list, r2_list, marker='o', color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (R^2)")
plt.title("Epoch vs Accuracy (R^2)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("epoch_vs_accuracy.png", dpi=150, bbox_inches='tight')
print("Saved: epoch_vs_accuracy.png")
plt.show()

# 6b) Epoch vs Loss (MSE)
plt.figure(figsize=(6,4))
plt.plot(epoch_list, mse_list, marker='o', color='purple')
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Epoch vs Loss (MSE)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("epoch_vs_loss.png", dpi=150, bbox_inches='tight')
print("Saved: epoch_vs_loss.png")
plt.show()

# 6c) Scatter + Regression Line
# Sort by area for a clean line plot
sort_idx = np.argsort(X_raw[:, 0])
X_sorted = X_raw[sort_idx]
y_line = y_pred_train[sort_idx]

plt.figure(figsize=(6,4))
plt.scatter(X_raw, y, color="blue", alpha=0.6, label="Data")
plt.plot(X_sorted, y_line, color="red", linewidth=2, label="Regression line")
plt.xlabel("Area (sq. ft.)")
plt.ylabel("Price")
plt.title("Linear Regression: Price vs Area")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("price_vs_area_regression.png", dpi=150, bbox_inches='tight')
print("Saved: price_vs_area_regression.png")
plt.show()