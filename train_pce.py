import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import UQpy as uq

# import logging  # Optional, display UQpy logs
# logger = logging.getLogger("UQpy")
# logger.setLevel(logging.INFO)

plt.style.use(["ggplot", "surg.mplstyle"])

# load data
filename = "./data/aerodynamics_data_100.csv"
df = pd.read_csv(filename, index_col=0)
input_columns = df.columns[0:5]
output_columns = df.columns[5:9]
n_samples = len(df)

# split into training and testing data sets
split = int(n_samples * 0.5)
x = df[input_columns].to_numpy()
y = df[output_columns].to_numpy()
x_train = x[0:split]
y_train = y[0:split]
x_test = x[split:]
y_test = y[split:]

# define joint distribution of inputs
marginals = [
    uq.Normal(0.0, 0.1),
    uq.Normal(500_000, 2_500),
    uq.Normal(0.3, 0.015),
    uq.Normal(0.7, 0.021),
    uq.Normal(0, 0.08),
]
distribution = uq.JointIndependent(marginals)

# define and fit Polynomial Chaos Expansion
polynomial_basis = uq.TotalDegreeBasis(
    distribution, max_degree=2
)  # ToDo: with least angle regression, try max_degree 4 or 5
regression_method = (
    uq.LeastSquareRegression()
)  # ToDo: replace with least angle regression
pce = uq.PolynomialChaosExpansion(polynomial_basis, regression_method)
pce.fit(x_train, y_train)

# compute PCE predictions, moments, and Sobol indices
pce_prediction = pce.predict(x_test)
pce_mean, pce_variance = pce.get_moments(higher=False)
pce_sensitivity = uq.PceSensitivity(pce)
start = time.time()
pce_sensitivity.run()
stop = time.time()
pce_sensitivity_time = stop - start

# compute Monte Carlo Sobol indices
uqpy_model = uq.RunModel(
    uq.PythonModel(
        model_script="model.py",
        model_object_name="xfoil",
    )
)
mc_sensitivity = uq.SobolSensitivity(uqpy_model, distribution)
start = time.time()
mc_sensitivity.run(
    n_samples=10
)  # ToDo: how many samples should we use? We're limited by the speed of XFOIL model
stop = time.time()
mc_sensitivity_time = stop - start

# compare moments from PCE and Monte Carlo
df = pd.read_csv("./data/aerodynamics_data_1000.csv", index_col=0)
print("Lift Statistics".ljust(17), "Mean".ljust(7), "Variance")
print("PCE".ljust(17), f"{float(pce_mean[0]): 1.4f} {float(pce_variance[0]):1.4e}")
print(
    "Monte Carlo".ljust(17),
    f"{df['Lift (CL)'].mean(): 1.4f} {df['Lift (CL)'].var():1.4e}",
)
# print("Torque Statistics".ljust(17), "Mean".ljust(7), "Variance")
print("Torque Statistics")
print("PCE".ljust(17), f"{float(pce_mean[3]): 1.4f} {float(pce_variance[3]):1.4e}")
print(
    "Monte Carlo".ljust(17),
    f"{df['Torque (CM)'].mean(): 1.4f} {df['Torque (CM)'].var():1.4e}",
)
print("")

# compare Sobol indices from PCE and Monte Carlo
print(f"Computed Sobol indices using MC in {mc_sensitivity_time:.4f} seconds")
print("MC Sobol Indices".ljust(27), "Lift".ljust(6), "Torque")
for i in range(len(input_columns)):
    sobol_lift = mc_sensitivity.first_order_indices[i, 0]
    sobol_torque = mc_sensitivity.first_order_indices[i, 3]
    print(input_columns[i].ljust(27), f"{sobol_lift: .2f} {sobol_torque: 6.2f}")

print(f"Computed Sobol indices using PCE in {pce_sensitivity_time:.4f} seconds")
print("PCE Sobol Indices".ljust(27), "Lift".ljust(6), "Torque")
for i in range(len(input_columns)):
    sobol_lift = pce_sensitivity.first_order_indices[i, 0]
    sobol_torque = pce_sensitivity.first_order_indices[i, 3]
    print(input_columns[i].ljust(27), f"{sobol_lift: .2f} {sobol_torque: 6.2f}")

# plot predictions and results
mse_error = np.linalg.norm(pce_prediction - y_test, axis=1)
fig, ax = plt.subplots()
ax.hist(mse_error, bins=30, edgecolor=ax.get_facecolor())
ax.set_title("Histogram of Test Data Error")
ax.set(xlabel="MSE $|| y - \hat{y} ||$", ylabel="Log of Counts")
ax.set_yscale("log")
fig.tight_layout()

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))
ax0.scatter(
    x_test[:, 0],
    y_test[:, 0],
    label="XFoil Lift",
    color="tab:gray",
    marker="o",
    s=10**2,
    alpha=0.5,
)
ax0.scatter(
    x_test[:, 0],
    pce_prediction[:, 0],
    label="PCE Lift",
    color="black",
    marker="+",
    s=6**2,
)
ax0.set_title("PCE Predictions for Lift")
ax0.set(xlabel="Angle of Attack", ylabel="Lift (CL)")
ax0.legend(loc="lower right")

ax1.scatter(
    x_test[:, 0],
    y_test[:, 3],
    label="XFoil Torque",
    color="tab:gray",
    marker="o",
    s=10**2,
    alpha=0.5,
)
ax1.scatter(
    x_test[:, 0],
    pce_prediction[:, 3],
    label="PCE Torque",
    color="black",
    marker="+",
    s=6**2,
)
ax1.set_title("PCE Predictions for Torque")
ax1.set(xlabel="Angle of Attack", ylabel="Torque (CM)")
ax1.legend(loc="lower right")
fig.tight_layout()

plt.show()
