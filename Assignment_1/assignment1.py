
import pandas as pd
import numpy as np
train=pd.read_csv('FMLA1Q1Data_train.csv',header=None)
test=pd.read_csv('FMLA1Q1Data_test.csv',header=None)
train.rename(columns={0: 'feature1', 1: 'feature2',2:'y'}, inplace=True)
test.rename(columns={0: 'feature1', 1: 'feature2',2:'y'}, inplace=True)

X = train.iloc[:, :-1].values  # Feature matrix
y = train.iloc[:, -1].values   # Target vector
X_train=X
y_train=y
# Adding intercept term (bias) to X
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Calculate the coefficients using the normal equation
cov_X_train = np.matmul(X.T, X)
inv_cov=np.linalg.inv(cov_X_train)
Xy=np.matmul(X.T,y)

theta=np.matmul(inv_cov,Xy)
wML=theta
# Predict y using the learned coefficients
y_pred = X @ theta

print("Coefficients (theta):", theta)


import numpy as np
import matplotlib.pyplot as plt


# Gradient Descent Algorithm
def gradient_descent(X, y, learning_rate, num_iterations):
  m = len(y)
  w = np.zeros(X.shape[1])  # Initialize weights to zeros
  norm_history = []

  for i in range(num_iterations):
    y_pred = X @ w
    error = y_pred - y
    gradient = (1/m) * X.T @ error
    w = w - learning_rate * gradient
    norm = np.linalg.norm(w - theta)  # Calculate norm ∥wt − wML∥2
    norm_history.append(norm)

  return w, norm_history


# Set hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Run gradient descent
w_gd, norm_history = gradient_descent(X, y, learning_rate, num_iterations)

# Plot the norm ∥wt − wML∥2 as a function of t
plt.plot(range(num_iterations), norm_history)
plt.xlabel("Iterations (t)")
plt.ylabel("∥wt − wML∥2")
plt.title("Norm of Weight Difference vs. Iterations")
plt.show()

# Print the final weights obtained from gradient descent
print("Weights (w_gd) from Gradient Descent:", w_gd)


# Stochastic Gradient Descent with batch size 100
def stochastic_gradient_descent(X, y, learning_rate, num_iterations, batch_size):
  m = len(y)
  w = np.zeros(X.shape[1])  # Initialize weights to zeros
  cost_history = []
  norm_history = []
  w_history = [w]
  w_avg_list=[w]
  for i in range(num_iterations):
    # Randomly select a batch of data
    indices = np.random.choice(m, batch_size, replace=False)
    X_batch = X[indices]
    y_batch = y[indices]

    y_pred = X_batch @ w
    gradient = 2 * X_batch.T @ (X_batch @ w - y_batch)
    lr = learning_rate / (1 + decay * i)
    w = w - lr * gradient

    # Calculate cost and norm using the full dataset for comparison
    y_pred_full = X @ w
    error_full = y_pred_full - y
    cost = (1/(2*m)) * np.sum(error_full**2)
    cost_history.append(cost)
    w_history.append(w)
    w_avg=np.mean(w_history)
    w_avg_list.append(w_avg)
  return w_avg, w_avg_list
# Set hyperparameters for SGD
batch_size = 100
num_iterations = 500
learning_rate = 0.0004
decay = 0.01
# Run stochastic gradient descent
w_sgd, w_history_sgd = stochastic_gradient_descent(X, y, learning_rate, num_iterations, batch_size)

# Calculate ||wt - wML||^2 for each iteration
norm_diff_sgd = [np.linalg.norm(w_t - wML)**2 for w_t in w_history_sgd]

# Plot the norm ∥wt − wML∥2 as a function of t for SGD
plt.plot(norm_diff_sgd)
plt.xlabel("Iteration")
plt.ylabel("||wt - wML||^2")
plt.title("Stochastic Gradient Descent: ||wt - wML||^2 vs. Iteration")
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Assuming X_train and y_train are already defined as in the preceding code

# Calculate the least squares solution wML
wML = np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)
print(wML)
# Gradient Descent
def gradient_descent(X, y, w_init, learning_rate, num_iterations):
    w = w_init
    w_history = [w]
    for i in range(num_iterations):
        gradient = 2 * X.T @ (X @ w - y)
        w = w - learning_rate * gradient
        w_history.append(w)
    return w, w_history

# Initialize w
w_init = np.zeros(X_train.shape[1])

# Set learning rate and number of iterations
learning_rate = 0.000001
num_iterations = 1000

# Run gradient descent
w_gd, w_history_gd = gradient_descent(X_train, y_train, w_init, learning_rate, num_iterations)
print(w_gd)
# Calculate ||wt - wML||^2 for each iteration
norm_diff_gd = [np.linalg.norm(w_t - wML)**2 for w_t in w_history_gd]

# Plot the results
plt.plot(norm_diff_gd)
plt.xlabel("Iteration")
plt.ylabel("||wt - wML||^2")
plt.title("Gradient Descent: ||wt - wML||^2 vs. Iteration")
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the training data
data_train = pd.read_csv('FMLA1Q1Data_train.csv', header=None)
X_train = data_train.iloc[:, :-1].values
y_train = data_train.iloc[:, -1].values
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

# Load the test data
data_test = pd.read_csv('FMLA1Q1Data_test.csv', header=None)
X_test = data_test.iloc[:, :-1].values
y_test = data_test.iloc[:, -1].values
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Function for ridge regression with gradient descent
def ridge_gradient_descent(X, y, learning_rate, num_iterations, lmbda):
  m = len(y)
  w = np.zeros(X.shape[1])  # Initialize weights
  cost_history = []

  for i in range(num_iterations):
    y_pred = X @ w
    error = y_pred - y
    gradient = (1/m) * X.T @ error + (lmbda / m) * w  # Ridge regression gradient
    w = w - learning_rate * gradient
    cost = (1/(2*m)) * np.sum(error**2) + (lmbda/(2*m)) * np.sum(w**2) # Ridge cost function
    cost_history.append(cost)

  return w, cost_history

# Function to calculate mean squared error
def mse(y_true, y_pred):
  return np.mean((y_true - y_pred)**2)

# Cross-validation for different lambda values
lambdas = np.logspace(-5, 2, 20)  # Example range of lambda values
validation_errors = []
for lmbda in lambdas:

  w_ridge, _ = ridge_gradient_descent(X_train, y_train, 0.01, 1000, lmbda)
  y_val_pred = X_train @ w_ridge
  validation_error = mse(y_train, y_val_pred)
  validation_errors.append(validation_error)


# Plot validation errors against lambda
plt.plot(lambdas, validation_errors)
plt.xlabel("Lambda")
plt.ylabel("Validation Error")
plt.title("Validation Error vs. Lambda")
plt.xscale('log')
plt.show()


# Choose the best lambda (e.g., the one with the lowest validation error)
best_lambda = lambdas[np.argmin(validation_errors)]
print("Best Lambda:", best_lambda)

# Train the model with the best lambda on the full training data
w_ridge, _ = ridge_gradient_descent(X_train, y_train, 0.01, 1000, best_lambda)

# Calculate the test error for wR
y_test_pred_ridge = X_test @ w_ridge
test_error_ridge = mse(y_test, y_test_pred_ridge)

# Calculate the test error for wML (obtained from the previous code)
y_test_pred_ml = X_test @ theta  # Assuming you have 'theta' from the previous code
test_error_ml = mse(y_test, y_test_pred_ml)

# Compare test errors
print("Test Error (wR):", test_error_ridge)
print("Test Error (wML):", test_error_ml)

# Analyze which model is better and why
if test_error_ridge < test_error_ml:
  print("Ridge regression (wR) is better.")
  # Ridge regression might be better due to regularization, reducing overfitting.
else:
  print("Least squares (wML) is better or they are similar.")
  # wML might be better if the data has a strong linear relationship
  # or if regularization is not needed.

# Load the training data
data_train = pd.read_csv('FMLA1Q1Data_train.csv', header=None)
X_train = data_train.iloc[:, :-1].values
y_train = data_train.iloc[:, -1].values

# Load the test data
data_test = pd.read_csv('FMLA1Q1Data_test.csv', header=None)
X_test = data_test.iloc[:, :-1].values
y_test = data_test.iloc[:, -1].values

# Adding intercept term (bias) to X_train and X_test
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test)) # Add bias term to X_test

# Define a Gaussian kernel function
def gaussian_kernel(x1, x2, sigma=1.0):
  return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * sigma**2))

# Kernel Regression function
def kernel_regression(X_train, y_train, X_test, kernel, sigma=1.0):
  y_pred = []
  for x_test in X_test:
    weights = [kernel(x_test, x_train, sigma) for x_train in X_train]
    y_pred.append(np.sum(weights * y_train) / np.sum(weights))
  return np.array(y_pred)

# Predict using Gaussian kernel regression
sigma = 1.0  # You can tune this hyperparameter
y_pred_kernel = kernel_regression(X_train, y_train, X_test, gaussian_kernel, sigma)

# Calculate Mean Squared Error (MSE) for kernel regression
mse_kernel = np.mean((y_test - y_pred_kernel)**2)


theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train # Recalculate theta to ensure X_train has the bias term

y_test_pred_ml = X_test @ theta
mse_ml = np.mean((y_test - y_test_pred_ml)**2)

print("MSE (Kernel Regression):", mse_kernel)
print("MSE (Least Squares):", mse_ml)

# Compare test errors
if mse_kernel < mse_ml:
  print("Kernel Regression is better.")
  # Kernel regression might be better because it can capture non-linear patterns.
else:
  print("Least squares is better or they are similar.")
  # wML might be better if the data has a strong linear relationship
  # or if kernel regression is not needed.