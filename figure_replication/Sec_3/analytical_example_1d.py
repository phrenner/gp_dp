###############################################################################
#                                                                             #
#             MACHINE LEARNING FOR DYNAMIC INCENTIVE PROBLEMS                 #
#              Replication of Section 3.3: 1-D Analytical Example             #
#                                                                             #
#  Paper Title:   Machine Learning for Dynamic Incentive Problems             #
#  Authors:       Philipp Renner (Lancaster U) & Simon Scheidegger (UNIL)     #
#  Source:        Section 3.3, Figure 1                                       #
#                                                                             #
#  DESCRIPTION:                                                               #
#  This script replicates the first analytical example presented in the       #
#  paper. It demonstrates how Gaussian Process Regression (GPR) combined      #
#  with Bayesian Active Learning (BAL) can efficiently approximate a          #
#  non-linear function.                                                       #
#                                                                             #
#  DETAILS:                                                                   #
#  - Target Function: f(x) = x^(sin(x))                                       #
#  - Domain: x in [0, 5]                                                      #
#  - Initial Data: 5 specific points {0.1, 0.5, 4.3, 4.5, 4.9}                #
#  - Method:                                                                  #
#      1. Fit a GP to the initial baseline points.                            #
#      2. Use the BAL acquisition function (Eq. 10) to select                 #
#         new points where the model is most uncertain.                       #
#      3. Visualize the reduction in uncertainty (Figure 1).                  #
#                                                                             #
#  HYPERPARAMETER SELECTION (sigma_m, sigma_v):                               #
#  The acquisition function is defined as:                                    #
#       alpha(x) = sigma_m * |mu(x)| + (sigma_v / 2) * log(sigma(x))          #
#                                                                             #
#  1. sigma_m (Exploitation Weight):                                          #
#     - Controls how much the algorithm focuses on regions with high          #
#       predicted values (e.g., maximizing utility/value functions).          #
#     - In this example, we set sigma_m = 1.0.                                #
#                                                                             #
#  2. sigma_v (Exploration Weight):                                           #
#     - Controls the incentive to explore regions with high uncertainty       #
#       (high variance sigma).                                                #
#     - A higher sigma_v encourages the algorithm to pick points in           #
#       unexplored gaps (like the gap between x=0.5 and x=4.3).               #
#     - We use sigma_v = 4.0 to prioritize reducing the massive               #
#       uncertainty in the middle of the domain.                              #
#                                                                             #
#  COLOR CODING:                                                              #
#  - Red Dots: Baseline observations (Initial Training Set)                   #
#  - Green Diamonds: Points added via Bayesian Active Learning                #
#                                                                             #
#  OUTPUT:                                                                    #
#  - Saves the figure to 'figure_1_replication.pdf'                     #
#  - Prints the Mean Approximation Error (L1) at each step.                   #
#                                                                             #
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 1. Define the analytical target function from Section 3.3
def target_function(x):
    # f(x) = x^(sin(x))
    return x ** np.sin(x)

# 2. Define the BAL Acquisition Function 
def acquisition_function(gp, x_candidates, sigma_m=1.0, sigma_v=4.0):
    """
    Simple acquisition function: alpha(x) = sigma_m * mu(x) + (sigma_v / 2) * log(sigma(x))
    """
    mu, sigma = gp.predict(x_candidates, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    score = sigma_m * mu + (sigma_v / 2.0) * np.log(sigma)
    return score

# --- Setup ---
np.random.seed(42)

# Domain for plotting
X_plot = np.linspace(0, 5, 10000)[:, None]
y_plot = target_function(X_plot).ravel()

# Initial training data
X_initial = np.array([0.1, 0.5, 4.3, 4.5, 4.9])[:, None]
y_initial = target_function(X_initial).ravel()

X_train = X_initial.copy()
y_train = y_initial.copy()

# Simple RBF Kernel 
# We use a ConstantKernel (C) to scale the amplitude and RBF for the shape.
# We use standard bounds (1e-2 to 1e2) rather than the constrained ones.
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

# Initialize GP with random_state for replicability
# Note: We added alpha=1e-5 to handle numerical stability (replacing the WhiteKernel)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-5, random_state=42)

# --- Main Loop ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ["Initial GP (5 points)", "BAL Step 1 (+1 point)", "BAL Step 2 (+1 point)"]

n_initial = len(X_initial)

for i in range(3):
    ax = axes[i]
    
    # A. Fit GP
    gp.fit(X_train, y_train)
    
    # B. Predict
    y_pred, y_std = gp.predict(X_plot, return_std=True)
    
    # --- UPDATED: Calculate Mean Approximation Error (L1) ---
    mae = np.mean(np.abs(y_pred - y_plot))
    print(f"[{titles[i]}] Mean Approximation Error (L1): {mae:.6f}")
    
    # C. Plotting
    # True Function
    ax.plot(X_plot, y_plot, 'r--', label='f(x) = x$^{sin(x)}$', alpha=0.6)
    
    # GP Approximation
    ax.plot(X_plot, y_pred, 'b-', label='GP Prediction')
    
    # Confidence Intervals (The "Fat Belly")
    ax.fill_between(X_plot.ravel(), 
                    y_pred - 1.96 * y_std, 
                    y_pred + 1.96 * y_std, 
                    alpha=0.2, color='blue', label='95% CI')
    
    # Scatter: Baseline Points
    ax.scatter(X_train[:n_initial], y_train[:n_initial], 
               c='red', s=60, zorder=10, label='Baseline Observations')
    
    # Scatter: New Points
    if len(X_train) > n_initial:
        ax.scatter(X_train[n_initial:], y_train[n_initial:], 
                   c='lime', edgecolors='black', s=80, marker='D', zorder=11, 
                   label='BAL Added Points')
    
    ax.set_title(titles[i])
    
    # Formatting
    ax.set_xlabel('x', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$f(x) = x^{\sin(x)}$', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlim(0, 5)
    
    # --- MODIFIED: INCREASED LEGEND FONT SIZE ---
    if i == 0:
        ax.legend(loc='upper left', fontsize=12)

    # D. Bayesian Active Learning
    if i < 2:
        X_candidates = np.random.uniform(0, 5, 2000)[:, None]
        scores = acquisition_function(gp, X_candidates, sigma_m=1.0, sigma_v=4.0)
        
        best_idx = np.argmax(scores)
        new_x = X_candidates[best_idx][:, None]
        new_y = target_function(new_x).ravel()
        
        X_train = np.vstack([X_train, new_x])
        y_train = np.hstack([y_train, new_y])
        
        print(f"Step {i+1}: Added point at x ≈ {new_x.item():.4f}")

plt.tight_layout()
plt.savefig("figure_1_replication.pdf")
