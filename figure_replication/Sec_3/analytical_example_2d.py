###############################################################################
#                                                                             #
#             MACHINE LEARNING FOR DYNAMIC INCENTIVE PROBLEMS                 #
#             Replication of Section 3.3: 2-D Analytical Example              #
#                                                                             #
#  Paper Title:   Machine Learning for Dynamic Incentive Problems             #
#  Authors:       Philipp Renner (Lancaster U) & Simon Scheidegger (UNIL)     #
#  Source:        Section 3.3, Figure 2                                       #
#                                                                             #
#  DESCRIPTION:                                                               #
#  This script replicates the second analytical example (2D Gaussian Peak).   #
#  It compares the performance of Gaussian Process Regression (GPR) trained   #
#  via Bayesian Active Learning (BAL) vs. Random Sampling vs. Sparse Grids.   #
#                                                                             #
#  DETAILS:                                                                   #
#  - Target Function: f(x) = exp(-sum(a_i^2 * (x_i - u_i)^2))                 #
#  - Parameters: d=2, x in [0,1]^2, a=(5,5), u=(0.8, 0.8)                     #
#  - Initial Data: 100 random points                                          #
#  - Enhancement: Add 40 points (via BAL vs. Uniform Random)                  #
#  - Metric: Mean Approximation Error (L1)                                    #
#                                                                             #
#  OUTPUT:                                                                    #
#  - Saves the figure to 'figure_2_replication.pdf'                    #
#                                                                             #
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
import Tasmanian
import warnings

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- 1. Define Target Function ------------------------------------------------
def f(x):
    """
    f(x) = exp(-sum(a_i^2 * (x_i - u_i)^2))
    Represents the 2D Gaussian Peak function described in Section 3.3.
    """
    a = np.array([5.0, 5.0])
    u = np.array([0.8, 0.8])
    return np.exp(-np.sum(a**2 * (x - u)**2, axis=1))


# --- 2. Parameters ------------------------------------------------------------
domain = [0, 1]
initial_points = 100           # size of initial training set
add_points = 40                # FIXED: Changed from 20 to 40 to match paper
candidates_num = 2000          # candidate pool for BAL
test_points_num = 10000        # large test set for accurate error estimation

# BAL Hyperparameters (derived from Eq. 10)
sigma_m = 1.0
sigma_v = 4.0  # high variance weight to encourage exploration

# Kernel: start with length_scale=0.2 to match function "steepness" (a=5)
kernel = (
    ConstantKernel(1.0)
    * RBF(0.2, length_scale_bounds=(1e-2, 1e1))
    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-2))
)

# --- 3. Setup Data ------------------------------------------------------------
np.random.seed(42)

# Initial Training Set
X_initial = np.random.uniform(domain[0], domain[1], (initial_points, 2))
y_initial = f(X_initial)

# Candidate Set for BAL
X_candidates = np.random.uniform(domain[0], domain[1], (candidates_num, 2))

# Test Set for Error Calculation
X_test = np.random.uniform(domain[0], domain[1], (test_points_num, 2))
y_test = f(X_test)


# --- 4. Helper Functions ------------------------------------------------------
def compute_error_gp(gp, X_test, y_test):
    """Computes Mean Absolute Error (L1) for GP predictions."""
    y_pred = gp.predict(X_test)
    return np.mean(np.abs(y_test - y_pred))


def acquisition(X, gp, sigma_m, sigma_v):
    """
    Implements Eq (10): alpha(x) = sigma_m * |mu(x)| + (sigma_v / 2) * log(sigma(x))
    Uses absolute mean to target high magnitude regions.
    """
    mu, sigma = gp.predict(X, return_std=True)
    return sigma_m * np.abs(mu) + (sigma_v / 2.0) * np.log(sigma + 1e-10)


# --- 5. Run BAL Experiment ----------------------------------------------------
print("\n--- Starting Bayesian Active Learning (BAL) ---")

X_train_bal = X_initial.copy()
y_train_bal = y_initial.copy()
bal_added = []
errors_bal = []
sample_sizes_bal = []

gp_bal = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

# Initial Fit
gp_bal.fit(X_train_bal, y_train_bal)
errors_bal.append(compute_error_gp(gp_bal, X_test, y_test))
sample_sizes_bal.append(len(y_train_bal))
print(f"Initial BAL: size = {sample_sizes_bal[-1]}, error = {errors_bal[-1]:.6f}")

# Active Learning Loop
for i in range(add_points):
    # Select point with highest acquisition score
    acq_values = acquisition(X_candidates, gp_bal, sigma_m, sigma_v)
    best_idx = np.argmax(acq_values)
    new_x = X_candidates[best_idx]
    new_y = f(new_x.reshape(1, -1))[0]

    # Store
    bal_added.append(new_x)
    X_train_bal = np.vstack([X_train_bal, new_x])
    y_train_bal = np.append(y_train_bal, new_y)

    # Update Model
    gp_bal.fit(X_train_bal, y_train_bal)

    # Track Metrics
    current_error = compute_error_gp(gp_bal, X_test, y_test)
    errors_bal.append(current_error)
    sample_sizes_bal.append(len(y_train_bal))

    # Print Intermediate Result
    print(f"BAL added point {i+1}: size = {sample_sizes_bal[-1]}, error = {current_error:.6f}")


# --- 6. Run Uniform Experiment ------------------------------------------------
print("\n--- Starting Uniform Random Sampling ---")

X_train_uni = X_initial.copy()
y_train_uni = y_initial.copy()
uni_added = []
errors_uni = []
sample_sizes_uni = []

gp_uni = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

# Initial Fit
gp_uni.fit(X_train_uni, y_train_uni)
errors_uni.append(compute_error_gp(gp_uni, X_test, y_test))
sample_sizes_uni.append(len(y_train_uni))
print(f"Initial Uniform: size = {sample_sizes_uni[-1]}, error = {errors_uni[-1]:.6f}")

# Random Sampling Loop
for i in range(add_points):
    new_x = np.random.uniform(domain[0], domain[1], (1, 2))[0]
    new_y = f(new_x.reshape(1, -1))[0]

    uni_added.append(new_x)
    X_train_uni = np.vstack([X_train_uni, new_x])
    y_train_uni = np.append(y_train_uni, new_y)

    gp_uni.fit(X_train_uni, y_train_uni)

    current_error = compute_error_gp(gp_uni, X_test, y_test)
    errors_uni.append(current_error)
    sample_sizes_uni.append(len(y_train_uni))

    # Print Intermediate Result
    print(
        f"Uniform added point {i+1}: "
        f"size = {sample_sizes_uni[-1]}, error = {current_error:.6f}"
    )


# --- 7. Run ASG Experiment (Tasmanian) ---------------------------------------
print("\n--- Starting Adaptive Sparse Grid (ASG) ---")

# Setup TASMANIAN Grid
# "localp" rule is standard for adaptive interpolation.
# dim=2, outputs=1, depth=3 (start at level 3), order=1, rule="localp"
# CHECK: depth=3 matches requirement "start with an adaptive sparse grid of level 3"
grid_asg = Tasmanian.SparseGrid()
grid_asg.makeLocalPolynomialGrid(2, 1, 3, 1, "localp")

# FIX: Map domain to [0, 1]^2 using a single 2D array
domain_transform = np.array([[0.0, 1.0], [0.0, 1.0]])
grid_asg.setDomainTransform(domain_transform)

# Refinement settings
fTolerance = 1.0e-6  # Refinement threshold as in the paper
asg_points = []
asg_errors = []

def get_asg_values(points):
    # Vectorized evaluation of f(x) for Tasmanian
    return f(points).reshape(-1, 1)

# Helper to compute error for ASG
def compute_error_asg_grid(grid, X_test, y_test):
    y_pred = grid.evaluateBatch(X_test)
    return np.mean(np.abs(y_test - y_pred.flatten()))

# Initial Load
aPoints = grid_asg.getNeededPoints()
aValues = get_asg_values(aPoints)
grid_asg.loadNeededPoints(aValues)

# Compute Initial Error
err = compute_error_asg_grid(grid_asg, X_test, y_test)
asg_points.append(grid_asg.getNumLoaded())
asg_errors.append(err)
print(f"Initial ASG: size = {asg_points[-1]}, error = {asg_errors[-1]:.6f}")

# Refinement Loop (approx 4 steps to get levels 3->7)
# CHECK: Loop increments size by setting surplus refinement
for ref in range(5):
    # Refine grid based on surplus error
    grid_asg.setSurplusRefinement(fTolerance, -1, "classic")
    
    aPoints = grid_asg.getNeededPoints()
    # If no new points needed, convergence reached
    if aPoints.shape[0] == 0:
        print("ASG Converged (no new points needed).")
        break
        
    # Evaluate new points
    aValues = get_asg_values(aPoints)
    grid_asg.loadNeededPoints(aValues)
    
    # Track Metrics
    current_error = compute_error_asg_grid(grid_asg, X_test, y_test)
    
    asg_points.append(grid_asg.getNumLoaded())
    asg_errors.append(current_error)
    
    # Print Intermediate Result
    print(f"ASG refinement {ref+1}: size = {asg_points[-1]}, error = {current_error:.6f}")


# --- 8. Plotting with Styled Fonts and Labels --------------------------------
fig = plt.figure(figsize=(14, 10))

LABEL_FONT = {"fontsize": 14, "fontweight": "bold"}
LEGEND_SIZE = 12

# Top Left: Test Function ------------------------------------------------------
ax1 = fig.add_subplot(2, 2, 1, projection="3d")
x_grid = np.linspace(0, 1, 50)
y_grid = np.linspace(0, 1, 50)
X_g, Y_g = np.meshgrid(x_grid, y_grid)
Z_g = f(np.column_stack([X_g.ravel(), Y_g.ravel()])).reshape(X_g.shape)

ax1.plot_surface(X_g, Y_g, Z_g, cmap="viridis", alpha=0.8)

# Labels as in the paper
#ax1.set_title("f(x,y)", **LABEL_FONT)
ax1.set_xlabel("x", **LABEL_FONT)
ax1.set_ylabel("y", **LABEL_FONT)

# FIX: Disable auto-rotation to enforce custom rotation
ax1.zaxis.set_rotate_label(False) 
ax1.set_zlabel("f(x,y)", rotation=90, **LABEL_FONT)

# Rotate view to resemble paper figure
ax1.view_init(elev=30, azim=-135)

# Top Right: Locations of Added Points ----------------------------------------
ax2 = fig.add_subplot(2, 2, 2)

bal_added = np.array(bal_added)
uni_added = np.array(uni_added)

# Uniform added points (gold / orange)
if len(uni_added) > 0:
    ax2.scatter(
        uni_added[:, 0],
        uni_added[:, 1],
        c="gold",
        marker=".",
        s=100,
        edgecolors="orange",
        label="uniform",
    )

# Baseline points (initial training set)
# STYLE UPDATE: Changed from White/Blue to RED to match 1D Example
ax2.scatter(
    X_initial[:, 0],
    X_initial[:, 1],
    c="red",               # Updated Color
    edgecolors="none",     # Updated Edge
    marker="o",
    s=40,
    label="baseline",
)

# BAL added points
if len(bal_added) > 0:
    n_half = 20 # Fixed split for legend consistency (20 + 20)
    
    # First 20 Points
    if len(bal_added) >= n_half:
        ax2.scatter(
            bal_added[:n_half, 0],
            bal_added[:n_half, 1],
            c="lime",          # Matches 1D Example
            marker="D",        # Matches 1D Example
            s=60,
            edgecolors="black",
            label="BAL, points 1–20",
        )
    # Next 20 Points (Different marker to distinguish batch, same color style)
    if len(bal_added) > n_half:
        ax2.scatter(
            bal_added[n_half:, 0],
            bal_added[n_half:, 1],
            c="lime",          # Matches 1D Example
            marker="s",        
            s=60,
            edgecolors="black",
            label="BAL, points 21–40",
        )

# Axes labels as in the paper
ax2.set_xlabel("x", **LABEL_FONT)

# FIX: Force horizontal rotation (0 degrees) for the Y-label
ax2.set_ylabel("y", rotation=90, labelpad=15, **LABEL_FONT)

#ax2.set_title("Locations of added points", **LABEL_FONT)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.legend(fontsize=LEGEND_SIZE, loc="upper right")

# Bottom Left: Convergence (BAL vs Uniform) -----------------------------------
ax3 = fig.add_subplot(2, 2, 3)
ax3.semilogy(sample_sizes_bal, errors_bal, "b-o", linewidth=2, label="BAL")
ax3.semilogy(sample_sizes_uni, errors_uni, "r--x", linewidth=2, label="uniform")

#ax3.set_title("Convergence: BAL vs uniform", **LABEL_FONT)
ax3.set_xlabel("# Points", **LABEL_FONT)
ax3.set_ylabel("Mean approximation error", **LABEL_FONT)

# FIX: Force x-axis to start at 100
ax3.set_xlim(left=100)

ax3.legend(fontsize=LEGEND_SIZE)
ax3.grid(True, which="both", ls="--", alpha=0.4)
ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

# Bottom Right: All Methods ----------------------------------------------------
ax4 = fig.add_subplot(2, 2, 4)
ax4.loglog(sample_sizes_bal, errors_bal, "b-o", linewidth=2, label="BAL")
ax4.loglog(sample_sizes_uni, errors_uni, "r--x", linewidth=2, label="uniform")
ax4.loglog(asg_points, asg_errors, "k-s", linewidth=2, label="Adaptive sparse grid")

#ax4.set_title("Comparison with sparse grids", **LABEL_FONT)
ax4.set_xlabel("# Points", **LABEL_FONT)
ax4.set_ylabel("Mean approximation error", **LABEL_FONT)

# FIX: Add specific ticks and format them as scalars (e.g., 100, 200, 500...)
ax4.set_xticks([100, 200, 500, 1000, 2000])
ax4.get_xaxis().set_major_formatter(ScalarFormatter())

ax4.legend(fontsize=LEGEND_SIZE)
ax4.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig("figure_2_replication.pdf")
